import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/app.css';
import { App } from './App';
import { ProgressProvider } from './lib/progress';
import { LanguageProvider } from './lib/language';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ProgressProvider>
      <LanguageProvider>
        <App />
      </LanguageProvider>
    </ProgressProvider>
  </StrictMode>,
);
