import type { ComponentType } from 'react';
import type { Question } from '../components/Ask';

export interface Section {
  /** Stable id used in URLs and in the progress file, e.g. "5.3". */
  id: string;
  /** Shown in the rail: the book's section number, or blank for our own pages. */
  num: string;
  title: string;
  /** One sentence: what you will be able to do after it. */
  lede: string;
  /** Book pages this section draws on, or null when it is ours. */
  pages: string | null;
  component: ComponentType;
}

export interface Chapter {
  id: string;
  num: string;
  title: string;
  pages: string;
  sections: Section[];
  /** Every question the chapter can ask, by id: lessons use them and review recycles them. */
  questions: Record<string, Question>;
}

export type Tier = 'Deep' | 'Standard' | 'Condensed' | 'Skipped';
export interface PlanRow {
  num: string;
  title: string;
  pages: string;
  tier: Tier;
  why: string;
  milestone?: string;
  built?: boolean;
  ours?: boolean;
}
