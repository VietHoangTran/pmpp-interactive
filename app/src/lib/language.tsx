import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

type Language = 'en' | 'vi';
type LanguageChoice = {
  language: Language;
  setLanguage: (language: Language) => void;
};

const LanguageContext = createContext<LanguageChoice>({
  language: 'en',
  setLanguage: () => {},
});
const STORAGE_KEY = 'pmpp-language';

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'vi' ? 'vi' : 'en';
    } catch {
      return 'en';
    }
  });

  useEffect(() => {
    document.documentElement.lang = language;
    document.title = language === 'vi' ? 'PMPP – Học lập trình GPU' : 'PMPP, worked through';
    try {
      localStorage.setItem(STORAGE_KEY, language);
    } catch {
      // The choice still works for the current page if storage is unavailable.
    }
  }, [language]);

  return <LanguageContext.Provider value={{ language, setLanguage }}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const choice = useContext(LanguageContext);
  return { ...choice, t: (english: string, vietnamese: string) => choice.language === 'vi' ? vietnamese : english };
}
