import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './components/arc/foundation.css'
import './index.css'

const colorScheme = window.matchMedia('(prefers-color-scheme: dark)')
const syncArcTheme = () => { document.documentElement.dataset.theme = colorScheme.matches ? 'dark' : 'light' }
syncArcTheme()
colorScheme.addEventListener('change', syncArcTheme)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
