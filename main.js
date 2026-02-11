import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// CONFIGURATION
const CONFIG = {
    physics: {
        disturbanceInterval: 3.0, // Seconds between hits
        disturbanceStrength: 15.0,
        orbitStrength: 1.0,
        orbitSpeed: 0.2
    },
    visuals: {
        gain: 0.3, // Arrow visual scaling
        maxLimit: 2.0, // Max arrow length
        noiseGate: 0.01 // Minimum force to draw arrow
    }
};

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

// PID CONTROLLER
class PIDController {
    constructor() {
        // Tuning parameters
        this.params = { Kp: 1.0, Ki: 0.0, Kd: 1.0 };

        // State memory
        this.integralSum = new THREE.Vector3(0, 0, 0);
        this.prevError = new THREE.Vector3(0, 0, 0);

        // Pre-allocated "scratchpad" vectors (optimization)
        this._error = new THREE.Vector3();
        this._pTerm = new THREE.Vector3();
        this._iTerm = new THREE.Vector3();
        this._dTermRaw = new THREE.Vector3();
        this._dTerm = new THREE.Vector3();
        this._dFiltered = new THREE.Vector3();
        this._total = new THREE.Vector3();
    }

    update(setpoint, measured, dt) {
        if (dt <= 0) return this._zeros();

        // Calculate error (In-place)
        this._error.copy(setpoint).sub(measured);

        // Proportional term
        this._pTerm.copy(this._error).multiplyScalar(this.params.Kp);

        // Integral term
        this.integralSum.addScaledVector(this._error, dt);

        const limit = 5.0; 
        this.integralSum.clampLength(0, limit);
        this._iTerm.copy(this.integralSum).multiplyScalar(this.params.Ki);

        // Derivative term
        const safeDt = dt > 0 ? dt : 0.016; 
        this._dTermRaw.copy(this._error).sub(this.prevError).divideScalar(safeDt);
        this._dFiltered.lerp(this._dTermRaw, 0.1); 
        this._dTerm.copy(this._dFiltered).multiplyScalar(this.params.Kd);

        // Update memory
        this.prevError.copy(this._error);

        // Sum output (In-place)
        this._total.copy(this._pTerm).add(this._iTerm).add(this._dTerm);

        return { output: this._total, p: this._pTerm, i: this._iTerm, d: this._dTerm };
    }

    reset() {
        this._error.set(0, 0, 0);
        this.integralSum.set(0, 0, 0);
        this.prevError.set(0, 0, 0);
        this._dFiltered.set(0, 0, 0);
    }

    _zeros() {
        return { output: new THREE.Vector3(), p: new THREE.Vector3(), i: new THREE.Vector3(), d: new THREE.Vector3() };
    }
}

class PowerMeter {
    constructor() {
        this.totalEnergy = 0;
        this.totalTime = 0;
        this.avgPower = 0;
    }

    update(forceVector, dt) {
        if (dt <= 0) return 0;

        // Model: Power = Force^2 (Simulating I^2*R electrical loss)
        const instantPower = forceVector.lengthSq(); // .lengthSq() is faster than length()^2

        // Integrate Energy (Joules = Watts * Seconds)
        this.totalEnergy += instantPower * dt;
        this.totalTime += dt;
        
        // Running Average
        this.avgPower = this.totalEnergy / this.totalTime;

        return instantPower;
    }
}

// TELEMETRY HUD
class TelemetryHUD {
    constructor(camera) {
        this.camera = camera;
        
        // HUD group (top left)
        this.hudGroup = new THREE.Group();
        this.camera.add(this.hudGroup); // Attach to camera!

        // UI anchor (bottom left)
        this.uiAnchor = new THREE.Object3D();
        this.camera.add(this.uiAnchor);

        // Create visual elements
        this.arrows = {
            p: this._createArrow(0xff0000, 0), // Red
            i: this._createArrow(0x00ff00, 0.8), // Green
            d: this._createArrow(0x0000ff, 1.6) // Blue
        };

        this.graphs = {
            p: this._createGraph(0xff0000, 2.5),
            i: this._createGraph(0x00ff00, 2.5),
            d: this._createGraph(0x0000ff, 2.5)
        };

        this._createBorder();
        this.updateLayout(); // Initial placement
    }

    // Public API
    update(pidResult) {
        // Update arrows and graphs
        this._updateComponent(pidResult.p, this.arrows.p, this.graphs.p);
        this._updateComponent(pidResult.i, this.arrows.i, this.graphs.i);
        this._updateComponent(pidResult.d, this.arrows.d, this.graphs.d);

        // Sync HTML GUI position
        this._syncGUI();
    }

    updateLayout() {
        const dist = 10;
        const vFOV = THREE.MathUtils.degToRad(this.camera.fov);
        const height = 2 * Math.tan(vFOV / 2) * dist;
        const width = height * this.camera.aspect;

        const padding = 1;
        const hudHeight = 2.5; // Matches border height

        // Position HUD (top left)
        this.hudGroup.position.set(-width / 2 + padding, height / 2 - padding - hudHeight, -dist);

        // Position GUI anchor (bottom left)
        // Little extra padding so it doesn't touch the screen edge
        this.uiAnchor.position.set(-width / 2 + 0.8, -height / 2 + 0.8, -dist);
    }

    // Internals
    _createArrow(color, xOffset) {
        const arrow = new THREE.ArrowHelper(
            new THREE.Vector3(1, 0, 0),
            new THREE.Vector3(0, 0, 0),
            1,
            color
        );
        arrow.position.set(xOffset, 1, 0);

        this.hudGroup.add(arrow);
        return arrow;
    }

    _createGraph(color, xOffset) {
        const maxPoints = 100;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(maxPoints * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const material = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
        const mesh = new THREE.Line(geometry, material);
        mesh.position.set(xOffset, 0.5, 0);
        this.hudGroup.add(mesh);
        return { mesh, positions, maxPoints };
    }

    _createBorder() {
        const w = 6.5; const h = 3.0;
        const pts = [-0.5, -0.5, 0, w, -0.5, 0, w, h, 0, -0.5, h, 0, -0.5, -0.5, 0];
        const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xffffff });
        this.hudGroup.add(new THREE.Line(geo, mat));
    }

    _updateComponent(vector, arrow, graph) {
        const rawMag = vector.length();
        const displayVal = Math.min(rawMag * CONFIG.visuals.gain, CONFIG.visuals.maxLimit);

        // Graph shift logic
        const arr = graph.positions;
        for (let i = 0; i < graph.maxPoints - 1; i++) {
            arr[i * 3 + 1] = arr[(i + 1) * 3 + 1];
        }
        arr[(graph.maxPoints - 1) * 3 + 1] = displayVal; 
        
        for (let i = 0; i < graph.maxPoints; i++) {
            arr[i * 3] = (i / graph.maxPoints) * 3.5; // Stretch width
            arr[i * 3 + 2] = 0;
        }
        graph.mesh.geometry.attributes.position.needsUpdate = true;

        // Arrow logic
        if (rawMag > CONFIG.visuals.noiseGate) {
            arrow.visible = true;
            arrow.setDirection(vector.clone().normalize());
            arrow.setLength(displayVal);
        } else {
            arrow.visible = false;
        }
    }

    _syncGUI() {
        const screenPos = new THREE.Vector3();
        this.uiAnchor.getWorldPosition(screenPos);
        screenPos.project(this.camera); // Project to normalized device coordinates (-1 to 1)

        const x = (screenPos.x * 0.5 + 0.5) * window.innerWidth;
        const y = (-(screenPos.y * 0.5) + 0.5) * window.innerHeight;

        const hudDiv = document.getElementById('hud-controls');
        if (hudDiv) {
            // Pivot from bottom-left: 
            // We subtract the DIV's height from Y to push it UP from the anchor point
            hudDiv.style.transform = `translate(${x}px, ${y - hudDiv.offsetHeight}px)`;
        }
    }
}

// MAIN APP LOGIC
let scene, camera, renderer, cube;
let controls;
let pid, telemetry, powerMeter;
let orbitArrow, impulseArrow;
let timeSinceLastChange = 0;
let disturbance = new THREE.Vector3();

// Fuel
let fuel = 100;
const fuelConsumptionRate = 0.05;

// Physics state
let physics;
let lastTime = performance.now();

// Trail setup
const trailLength = 120; // How long the tail is
const trailGeometry = new THREE.BufferGeometry();
// Create a fixed array of size N * 3 (for x, y, z)
const trailPositions = new Float32Array(trailLength * 3); 
trailGeometry.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));

const trailMaterial = new THREE.LineBasicMaterial({
    color: 0xffffff, // Cyan (matches your engineering vibe)
    transparent: true,
    opacity: 0.5     // Faint
});

const trail = new THREE.Line(trailGeometry, trailMaterial);

// Trail timing for frame-rate independence
const trailUpdateInterval = 1 / 60; // Add a point every ~16.7ms (60 Hz)
let timeSinceLastTrailUpdate = 0;

function init() {
    // Setup Three.js
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 10;
    camera.position.y = 1;
    scene.add(camera); // important

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Setup orbit controls
    controls = new OrbitControls(camera, renderer.domElement);

    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    controls.target.set(0, 0, 0);

    controls.enablePan = false;

    // Setup cube
    const geometry = new THREE.BoxGeometry(1, 1, 1);
    // const geometry = new THREE.IcosahedronGeometry(1, 0);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true });
    cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // Setup grid
    const gridHelper = new THREE.GridHelper(100, 50, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Setup physics state
    physics = new PhysicsState();

    // Setup constant force arrow
    orbitArrow = new THREE.ArrowHelper(
        new THREE.Vector3(),
        new THREE.Vector3(),
        0,
        0x00ffff // Cyan
    );
    cube.add(orbitArrow);

    // Setup impulse arrow
    impulseArrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, 0),
        0,
        0xffff00 // Yellow
    );
    cube.add(impulseArrow);

    // Initialize systems
    pid = new PIDController();
    telemetry = new TelemetryHUD(camera);
    powerMeter = new PowerMeter();

    // Bind sliders
    bindSlider('sliderP', 'valP', (v) => pid.params.Kp = v);
    bindSlider('sliderI', 'valI', (v) => pid.params.Ki = v);
    bindSlider('sliderD', 'valD', (v) => pid.params.Kd = v);

    // Add trail to scene
    scene.add(trail);

    // Random impulse button
    document.getElementById('btn-random').addEventListener('click', () => {
        // Define the strength of the kick (Newton-seconds)
        const kickStrength = 20.0; 

        // Generate a random direction
        const impulse = new THREE.Vector3(
            (Math.random() - 0.5),
            (Math.random() - 0.5),
            (Math.random() - 0.5)
        ).normalize().multiplyScalar(kickStrength);

        // Apply the Impulse (Force * Time = Change in Momentum)
        // Since Mass = 1, Change in Velocity = Impulse
        physics.velocity.add(impulse);
    });

    // Initial kick
    triggerDisturbance();
}

function bindSlider(id, labelId, callback) {
    const el = document.getElementById(id);
    const label = document.getElementById(labelId);

    // Force the input to revert to the hardcoded 'value' attribute in the HTML
    // This overrides the browser's "Restore form state" cache.
    if (el.getAttribute('value') !== null) {
        el.value = el.getAttribute('value');
    }

    const sync = () => {
        const val = parseFloat(el.value);
        // Display with 2 decimals for 'I', 1 for others
        label.innerText = val.toFixed(id === 'sliderI' ? 2 : 1);
        callback(val);
    };

    // Use 'input' for real-time updates as you drag
    el.addEventListener('input', sync);
    
    sync(); // important, make class start with reset values
}

function syncArrowToForce(arrow, forceVector, color) {
    const magnitude = forceVector.length();
    
    // Only show if the force is significant
    if (magnitude > 0.1) {
        arrow.visible = true;
        
        // Get the direction
        const dir = forceVector.clone().normalize();
        
        // Adjust for cube rotation (World space -> Local space)
        const worldToLocal = cube.quaternion.clone().invert();
        dir.applyQuaternion(worldToLocal);
        
        // Update the arrow visuals
        arrow.setDirection(dir);
        arrow.setLength(magnitude * 0.5); 
    } else {
        arrow.visible = false;
    }
}

function triggerDisturbance() {
    disturbance.set(
        (Math.random() - 0.5) * CONFIG.physics.disturbanceStrength,
        (Math.random() - 0.5) * CONFIG.physics.disturbanceStrength,
        (Math.random() - 0.5) * CONFIG.physics.disturbanceStrength
    );
    timeSinceLastChange = 0;
}

function animate() {
    requestAnimationFrame(animate);

    controls.update();

    const currentTime = performance.now() / 1000;
    let dt = currentTime - lastTime;
    if (dt > 0.1 || dt <= 0) {
        dt = 1 / 60;
    }
    lastTime = currentTime;

    // Rotating disturbance
    const speedTheta = CONFIG.physics.orbitSpeed;
    const speedPhi = CONFIG.physics.orbitSpeed * 0.73;

    // Calculate spherical to cartesian coordinates
    const theta = currentTime * speedTheta;
    const phi = currentTime * speedPhi;

    const orbitX = Math.sin(phi) * Math.cos(theta);
    const orbitY = Math.cos(phi);
    const orbitZ = Math.sin(phi) * Math.sin(theta);

    let rotatingForce = new THREE.Vector3(orbitX, orbitY, orbitZ);
    rotatingForce.multiplyScalar(CONFIG.physics.orbitStrength);

    // Apply to physics
    physics.acceleration.add(rotatingForce)

    // Apply periodic disturbance
    timeSinceLastChange += dt;
    if (timeSinceLastChange > CONFIG.physics.disturbanceInterval) {
        triggerDisturbance();
    }
    physics.acceleration.add(disturbance);
    // Decay disturbance (impulse simulation)
    disturbance.multiplyScalar(0.95);

    // Rotating force arrow
    syncArrowToForce(orbitArrow, rotatingForce);

    // Impulse arrow
    syncArrowToForce(impulseArrow, disturbance);

    // PID control
    const result = pid.update(new THREE.Vector3(0,0,0), physics.position, dt);
    physics.acceleration.add(result.output);

    // Physics integration
    physics.update(dt);

    // Sync the view to the model
    cube.position.copy(physics.position);

    // Little rotation
    cube.rotation.x += 0.5 * dt; 
    cube.rotation.y += 0.3 * dt;

    // Update visuals
    telemetry.update(result);

    const instantWatts = powerMeter.update(result.output, dt);

    // 3. Update HUD Text (Efficiently)
    // Only update text every 10 frames to prevent flickering/lag
    if (renderer.info.render.frame % 10 === 0) {
        document.getElementById('val-inst').innerText = instantWatts.toFixed(1);
        document.getElementById('val-avg').innerText = powerMeter.avgPower.toFixed(1);
    }

    // Trail update - frame-rate independent
    timeSinceLastTrailUpdate += dt;
    if (timeSinceLastTrailUpdate >= trailUpdateInterval) {
        timeSinceLastTrailUpdate -= trailUpdateInterval;
        
        // Shift all points down by one slot (delete the oldest)
        // This is a high-speed memory operation
        trailPositions.copyWithin(0, 3); 

        // Set the last point to the cube's current position
        const i = (trailLength - 1) * 3;
        trailPositions[i] = cube.position.x;
        trailPositions[i+1] = cube.position.y;
        trailPositions[i+2] = cube.position.z;

        // Tell the GPU the data changed
        trail.geometry.attributes.position.needsUpdate = true;
    }

    // Render
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    telemetry.updateLayout();
});

init();
animate();