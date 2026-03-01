import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        // 1. The main entry point
        main: resolve(__dirname, 'index.html'),
        // 2. Your about page
        about: resolve(__dirname, 'about.html'),
        // 3. The new Mechatronics history page
        mech: resolve(__dirname, 'mech.html'),
      },
    },
  },
});