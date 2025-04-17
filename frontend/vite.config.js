import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  root: 'frontend',  // This tells Vite that your source files are inside the 'frontend' directory
  plugins: [react()],
  build: {
    outDir: 'frontend/dist',  // This specifies the build output folder to be 'frontend/dist'
  },
});
