import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],

  server: {
    port: 5173,
    proxy: {
      '/data': 'http://localhost:8000',
      '/api':  'http://localhost:8000'
    }
  },

  build: {
    // Output the production build into FastAPI's static folder
    outDir: '../app/static'
  }
})
