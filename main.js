import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// CONFIGURATION
const CONFIG = {
    physics: {
        disturbanceInterval: 3.0,
        disturbanceStrength: 15.0,
        orbitStrength: 1.0,
        orbitSpeed: 0.2
    },
    visuals: {
        gain: 0.3,
        maxLimit: 2.0,
        noiseGate: 0.01
    }
};

// --- PHYSICS STATE ---
class PhysicsState {
    constructor() {
        this.position = new THREE.Vector3();
        this.velocity = new THREE.Vector3();
        this.acceleration = new THREE.Vector3();
    }

    update(dt) {
        this.velocity.addScaledVector(this.acceleration, dt);
        this.position.addScaledVector(this.velocity, dt);
        this.acceleration.set(0, 0, 0);
    }
}

// --- PID CONTROLLER ---
class PIDController {
    constructor() {
        this.params = { Kp: 1.0, Ki: 0.0, Kd: 1.0 };
        this.integralSum = new THREE.Vector3(0, 0, 0);
        this.prevError = new THREE.Vector3(0, 0, 0);
        
        this._error = new THREE.Vector3();
        this._pTerm = new THREE.Vector3();
        this._iTerm = new THREE.Vector3();
        
        // Added for D-term smoothing
        this._dTermRaw = new THREE.Vector3();
        this._dFiltered = new THREE.Vector3();
        
        this._dTerm = new THREE.Vector3();
        this._total = new THREE.Vector3();
    }

    update(setpoint, measured, dt) {
        if (dt <= 0) return this._zeros();
        this._error.copy(setpoint).sub(measured);

        // P
        this._pTerm.copy(this._error).multiplyScalar(this.params.Kp);

        // I
        this.integralSum.addScaledVector(this._error, dt);
        this.integralSum.clampLength(0, 5.0);
        this._iTerm.copy(this.integralSum).multiplyScalar(this.params.Ki);

        // D (with low-pass filter)
        const safeDt = dt > 0 ? dt : 0.016; 
        
        // 1. Get raw rate of change
        this._dTermRaw.copy(this._error).sub(this.prevError).divideScalar(safeDt);
        
        // 2. Smooth it (lerp acts as an exponential moving average)
        this._dFiltered.lerp(this._dTermRaw, 0.1); 
        
        // 3. Apply Kd gain to the smoothed value
        this._dTerm.copy(this._dFiltered).multiplyScalar(this.params.Kd);
        
        this.prevError.copy(this._error);
        this._total.copy(this._pTerm).add(this._iTerm).add(this._dTerm);

        return { output: this._total, p: this._pTerm, i: this._iTerm, d: this._dTerm };
    }

    reset() {
        this.integralSum.set(0, 0, 0);
        this.prevError.set(0, 0, 0);
        this._dFiltered.set(0, 0, 0); // Reset filter state
    }

    _zeros() {
        return { output: new THREE.Vector3(), p: new THREE.Vector3(), i: new THREE.Vector3(), d: new THREE.Vector3() };
    }
}

// --- MPC CONTROLLER (FEEDFORWARD UPGRADE) ---
class MPCController {
    constructor() {
        this.horizon = 1.0; 
        this.maxForce = 20.0; 
        this.output = new THREE.Vector3();
        this.predictedPath = []; 
    }

    update(position, velocity, dt, feedforward = new THREE.Vector3()) {
        const T = Math.max(0.1, this.horizon);
        
        // 1. Ideal Force (F = ma)
        const term1 = position.clone();
        const term2 = velocity.clone().multiplyScalar(T);
        const idealNetForce = term1.add(term2).multiplyScalar(-2).divideScalar(T * T);

        // 2. Feedforward Cancellation
        const motorForce = idealNetForce.clone().sub(feedforward);

        // 3. Constraints
        if (motorForce.length() > this.maxForce) {
            motorForce.setLength(this.maxForce);
        }

        this.output.copy(motorForce);

        // 4. Prediction
        const actualResultantForce = this.output.clone().add(feedforward);
        this._simulateFuture(position, velocity, actualResultantForce, T);

        return { output: this.output };
    }

    _simulateFuture(startPos, startVel, netForce, T) {
        this.predictedPath = [];
        const p = startPos.clone();
        const v = startVel.clone();
        const dt = 0.05; 
        const steps = Math.ceil(T / dt);

        this.predictedPath.push(p.clone());
        for(let i=0; i<steps; i++) {
            v.addScaledVector(netForce, dt);
            p.addScaledVector(v, dt);
            this.predictedPath.push(p.clone());
        }
    }
}

// --- POWER METER ---
class PowerMeter {
    constructor() {
        this.totalEnergy = 0;
        this.totalTime = 0;
        this.avgPower = 0;
    }

    update(forceVector, dt) {
        if (dt <= 0) return 0;
        const instantPower = forceVector.lengthSq();
        this.totalEnergy += instantPower * dt;
        this.totalTime += dt;
        this.avgPower = this.totalEnergy / this.totalTime;
        return instantPower;
    }
}

// --- TELEMETRY HUD ---
class TelemetryHUD {
    constructor(camera) {
        this.camera = camera;
        this.hudGroup = new THREE.Group();
        this.camera.add(this.hudGroup);
        
        this.arrows = {
            p: this._createArrow(0xff0000, 0),
            i: this._createArrow(0x00ff00, 0.8),
            d: this._createArrow(0x0000ff, 1.6)
        };

        this.graphs = {
            r: this._createGraph(0xff0000, 3.0), 
            g: this._createGraph(0x00ff00, 3.0), 
            b: this._createGraph(0x0000ff, 3.0)  
        };
        
        this._createBorder();
        this.updateLayout();
    }

    update(result, mode, physics) {
        if (mode === 'PID') {
            this.graphs.r.visible = true;
            this.graphs.g.visible = true;
            this.graphs.b.visible = true;

            this._updateGraph(this.graphs.r, result.p.length());
            this._updateGraph(this.graphs.g, result.i.length());
            this._updateGraph(this.graphs.b, result.d.length());

            this._updateArrow(result.p, this.arrows.p);
            this._updateArrow(result.i, this.arrows.i);
            this._updateArrow(result.d, this.arrows.d);
        } else {
            this.arrows.p.visible = false;
            this.arrows.i.visible = false;
            this.arrows.d.visible = false;

            this.graphs.r.visible = false;
            this.graphs.b.visible = false;
            this.graphs.g.visible = true; 
            
            this._updateGraph(this.graphs.g, physics.position.length());
        }
    }

    updateLayout() {
        // Z-Depth: Keep HUD close (z=4) to avoid clipping
        const dist = 4; 
        
        const vFOV = THREE.MathUtils.degToRad(this.camera.fov);
        const height = 2 * Math.tan(vFOV / 2) * dist;
        const width = height * this.camera.aspect;
        
        const hudW = 7.5; // Physical width of the HUD mesh
        
        let targetScale;
        
        if (this.camera.aspect < 1.0) {
            // PORTRAIT (Mobile)
            // Scale to fit 85% of screen width
            targetScale = (width * 0.85) / hudW;
        } else {
            // LANDSCAPE (Desktop/Tablet)
            // Aim for 25% of screen width
            let idealDesktop = (width * 0.25) / hudW;
            
            // HARD CLAMP: Never go above 0.5 scale on desktop
            targetScale = Math.min(idealDesktop, 0.5);
            
            // Safety floor: Don't let it get microscopic
            targetScale = Math.max(targetScale, 0.25);
        }

        this.hudGroup.scale.set(targetScale, targetScale, targetScale);

        // Position: Top-Left with padding
        const padding = 0.5 * targetScale;
        const localLeft = -0.5;
        const localTop = 2.5;

        this.hudGroup.position.set(
            -width / 2 + padding - (localLeft * targetScale), 
            height / 2 - padding - (localTop * targetScale), 
            -dist
        );
    }

    _createArrow(color, x) {
        const arrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), 1, color);
        arrow.position.set(x, 0.5, 0); 
        this.hudGroup.add(arrow);
        return arrow;
    }

    _createGraph(color, xOffset) {
        const maxPoints = 100;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(maxPoints * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const material = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
        const line = new THREE.Line(geometry, material);
        
        line.position.set(xOffset, 0, 0);
        line.userData = { points: positions, max: maxPoints };
        this.hudGroup.add(line);
        return line;
    }

    _updateGraph(lineObj, value) {
        if (!lineObj.visible) return;

        const positions = lineObj.userData.points;
        const max = lineObj.userData.max;

        for (let i = 0; i < max - 1; i++) {
            positions[i * 3 + 1] = positions[(i + 1) * 3 + 1];
        }

        let displayVal = value * 0.5; 
        if (displayVal > 2.0) displayVal = 2.0; 
        positions[(max - 1) * 3 + 1] = displayVal;

        for (let i = 0; i < max; i++) {
            positions[i * 3] = (i / max) * 3.5; 
            positions[i * 3 + 2] = 0;
        }

        lineObj.geometry.attributes.position.needsUpdate = true;
    }

    _createBorder() {
        const w = 7.0; 
        const h = 2.5;
        const pts = [-0.5,-0.5,0, w,-0.5,0, w,h,0, -0.5,h,0, -0.5,-0.5,0];
        const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
        this.hudGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffffff })));
    }

    _updateArrow(vec, arrow) {
        if (!vec) return;
        const mag = Math.min(vec.length() * CONFIG.visuals.gain, CONFIG.visuals.maxLimit);
        if(mag > CONFIG.visuals.noiseGate) {
            arrow.visible = true;
            arrow.setDirection(vec.clone().normalize());
            arrow.setLength(mag);
        } else {
            arrow.visible = false;
        }
    }
}

// --- MAIN APP ---
let scene, camera, renderer, droneMesh, controls;
let pid, mpc, telemetry, physics, powerMeter;
let orbitArrow, impulseArrow, predictedLine; 
let currentMode = 'PID';
let currentModel = 'CUBE';
let mikuVisual, cubeVisual;

// State
let lastTime = performance.now();
let timeSinceLastChange = 0;
let disturbance = new THREE.Vector3();

// Trail
const trailLength = 120;
const trailPositions = new Float32Array(trailLength * 3);
let trail;

function init() {
    scene = new THREE.Scene();

    // soft white light everywhere so she isn't completely swallowed by the void
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.0); 
    scene.add(ambientLight);

    // a strong directional light to give the 3d geometry some actual depth
    const directionalLight = new THREE.DirectionalLight(0xffffff, 3.0);
    directionalLight.position.set(5, 10, 7);
    scene.add(directionalLight);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1, 10);
    scene.add(camera);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    // Objects

    // 1. Create the permanent physics container immediately
    droneMesh = new THREE.Group();
    scene.add(droneMesh);

    // 2. Create the classic wireframe cube and put it in the container
    cubeVisual = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1), 
        new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true })
    );
    cubeVisual.visible = true; // cube is the default
    droneMesh.add(cubeVisual);

    // 3. Load the visual miku mesh asynchronously
    const loader = new GLTFLoader();
    loader.load(
        'models/hatsune_miku_plushie.glb',
        function (gltf) {
            mikuVisual = gltf.scene;
            mikuVisual.scale.set(2, 2, 2); 
            mikuVisual.position.set(0, -0.5, 0);
            mikuVisual.visible = false; // start hidden until user toggles
            droneMesh.add(mikuVisual);
        },
        undefined, 
        function (error) {
            console.error('failed to load miku:', error);
        }
    );

    scene.add(new THREE.GridHelper(100, 50, 0x444444, 0x222222));

    // Arrows - these now safely attach to the empty Group immediately
    orbitArrow = new THREE.ArrowHelper(new THREE.Vector3(0,1,0), new THREE.Vector3(), 0, 0x00ffff);
    impulseArrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(), 0, 0xffff00);
    droneMesh.add(orbitArrow);
    droneMesh.add(impulseArrow);

    // MPC Line
    const lineGeo = new THREE.BufferGeometry();
    const linePos = new Float32Array(200 * 3); 
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    predictedLine = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x00ff00 }));
    scene.add(predictedLine);

    // Trail
    const tGeo = new THREE.BufferGeometry();
    tGeo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
    trail = new THREE.Line(tGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 }));
    scene.add(trail);

    // Systems
    physics = new PhysicsState();
    pid = new PIDController();
    mpc = new MPCController(); 
    telemetry = new TelemetryHUD(camera);
    powerMeter = new PowerMeter();

    setupUI();
    triggerDisturbance();
    animate();
}

function setupUI() {
    const bind = (id, labelId, cb) => {
        const el = document.getElementById(id);
        const label = document.getElementById(labelId);
        if (!el) return;
        el.addEventListener('input', () => {
            const v = parseFloat(el.value);
            label.innerText = v.toFixed(id.includes('frc') ? 0 : 1);
            cb(v);
        });
    };

    bind('pid-p', 'val-pid-p', (v) => pid.params.Kp = v);
    bind('pid-i', 'val-pid-i', (v) => pid.params.Ki = v);
    bind('pid-d', 'val-pid-d', (v) => pid.params.Kd = v);

    bind('mpc-hz', 'val-mpc-hz', (v) => mpc.horizon = v);
    bind('mpc-frc', 'val-mpc-frc', (v) => mpc.maxForce = v);

    const toggleBtn = document.getElementById('btn-toggle');
    const hud = document.getElementById('hud-controls');
    const pidPanel = document.getElementById('pid-panel');
    const mpcPanel = document.getElementById('mpc-panel');
    const hrs = document.querySelectorAll('hr');

    toggleBtn.addEventListener('click', () => {
        currentMode = currentMode === 'PID' ? 'MPC' : 'PID';
        const isMPC = currentMode === 'MPC';

        pidPanel.style.display = isMPC ? 'none' : 'block';
        mpcPanel.style.display = isMPC ? 'block' : 'none';

        hud.classList.toggle('mpc-mode', isMPC);
        
        toggleBtn.innerText = `MODE: ${currentMode}`;
        toggleBtn.style.color = isMPC ? '#00ff00' : 'white';
        toggleBtn.style.borderColor = isMPC ? '#00ff00' : 'white';

        hrs.forEach(hr => hr.style.borderColor = isMPC ? '#004400' : '#444');

        pid.reset();
    });

    document.getElementById('btn-random').addEventListener('click', () => {
        const kick = new THREE.Vector3((Math.random()-0.5), (Math.random()-0.5), (Math.random()-0.5));
        physics.velocity.add(kick.normalize().multiplyScalar(20.0));
    });

    const btnModel = document.getElementById('btn-model');
    if (btnModel) {
        btnModel.addEventListener('click', () => {
            currentModel = currentModel === 'MIKU' ? 'CUBE' : 'MIKU';
            const isMiku = currentModel === 'MIKU';

            // toggle visibility safely (miku might still be downloading if you click too fast on a slow network)
            if (mikuVisual) mikuVisual.visible = isMiku;
            if (cubeVisual) cubeVisual.visible = !isMiku;

            // update button styling
            btnModel.innerText = `MDL: ${currentModel}`;
            btnModel.style.color = isMiku ? 'cyan' : 'white';
            btnModel.style.borderColor = isMiku ? 'cyan' : 'white';
        });
    }
}

function triggerDisturbance() {
    disturbance.set(
        (Math.random()-0.5)*CONFIG.physics.disturbanceStrength,
        (Math.random()-0.5)*CONFIG.physics.disturbanceStrength,
        (Math.random()-0.5)*CONFIG.physics.disturbanceStrength
    );
    timeSinceLastChange = 0;
}

function updateTrail() {
    trailPositions.copyWithin(0, 3);
    const i = (trailLength - 1) * 3;
    trailPositions[i] = droneMesh.position.x;
    trailPositions[i+1] = droneMesh.position.y;
    trailPositions[i+2] = droneMesh.position.z;
    trail.geometry.attributes.position.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);
    const now = performance.now() / 1000;
    let dt = now - lastTime;
    if (dt > 0.1 || dt <= 0) dt = 1/60;
    lastTime = now;

    controls.update();

    const theta = now * CONFIG.physics.orbitSpeed;
    const phi = now * CONFIG.physics.orbitSpeed * 0.73;
    const orbit = new THREE.Vector3(Math.sin(phi)*Math.cos(theta), Math.cos(phi), Math.sin(phi)*Math.sin(theta));
    orbit.multiplyScalar(CONFIG.physics.orbitStrength);
    
    physics.acceleration.add(orbit);
    
    timeSinceLastChange += dt;
    if (timeSinceLastChange > CONFIG.physics.disturbanceInterval) triggerDisturbance();
    physics.acceleration.add(disturbance);
    disturbance.multiplyScalar(0.95);

    let controlOut;
    
    if (currentMode === 'MPC') {
        const externalForces = new THREE.Vector3().copy(orbit).add(disturbance);
        controlOut = mpc.update(physics.position, physics.velocity, dt, externalForces);
        
        const path = mpc.predictedPath;
        const posAttr = predictedLine.geometry.attributes.position;
        let idx = 0;
        for (let i=0; i<path.length && idx < posAttr.count*3; i++) {
            posAttr.array[idx++] = path[i].x;
            posAttr.array[idx++] = path[i].y;
            posAttr.array[idx++] = path[i].z;
        }
        // update the vertices
        for (let i = 0; i < path.length && idx < posAttr.count * 3; i++) {
            posAttr.array[idx++] = path[i].x;
            posAttr.array[idx++] = path[i].y;
            posAttr.array[idx++] = path[i].z;
        }

        // tell three.js exactly how many vertices to draw
        predictedLine.geometry.setDrawRange(0, path.length);
        posAttr.needsUpdate = true;
        predictedLine.visible = true;

    } else {
        controlOut = pid.update(new THREE.Vector3(0,0,0), physics.position, dt);
        predictedLine.visible = false;
    }

    physics.acceleration.add(controlOut.output);
    physics.update(dt);

    droneMesh.position.copy(physics.position);
    // droneMesh.rotation.x += 0.5 * dt;
    droneMesh.rotation.y += 0.3 * dt;

    telemetry.update(controlOut, currentMode, physics);
    
    const orbMag = orbit.length();
    orbitArrow.visible = orbMag > 0.1;
    if (orbitArrow.visible) {
        const d = orbit.clone().normalize().applyQuaternion(droneMesh.quaternion.clone().invert());
        orbitArrow.setDirection(d);
        orbitArrow.setLength(orbMag * 0.5);
    }

    const impMag = disturbance.length();
    impulseArrow.visible = impMag > 0.1;
    if (impulseArrow.visible) {
        const d = disturbance.clone().normalize().applyQuaternion(droneMesh.quaternion.clone().invert());
        impulseArrow.setDirection(d);
        impulseArrow.setLength(impMag * 0.5);
    }

    updateTrail();

    const watts = powerMeter.update(controlOut.output, dt);
    if (renderer.info.render.frame % 10 === 0) {
        document.getElementById('val-inst').innerText = watts.toFixed(1);
        document.getElementById('val-avg').innerText = powerMeter.avgPower.toFixed(1);
    }

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    telemetry.updateLayout();
});

init();