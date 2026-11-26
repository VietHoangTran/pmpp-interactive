import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../components/Ask';
import { useProgress } from '../lib/progress';
import { INTERVAL_DAYS, isDue } from '../lib/review';
import { loadAllQuestions } from '../chapters/registry';
import { useLanguage } from '../lib/language';

// Questions you have met come back here on a widening schedule.
export function Review() {
  const { progress, loaded } = useProgress();
  const { t } = useLanguage();
  const [questions, setQuestions] = useState<Record<string, Question>>();
  useEffect(() => {
    loadAllQuestions().then(setQuestions);
  }, []);
  // Freeze the queue once everything is loaded, so answering does not reshuffle it.
  const ready = loaded && !!questions;
  const queue = useMemo(() => {
    if (!ready) return [];
    return Object.keys(progress.review)
      .filter((id) => questions![id] && isDue(progress.review[id]))
      .sort((a, b) => (progress.review[a].due < progress.review[b].due ? -1 : 1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);
  const [pos, setPos] = useState(0);
  const [answered, setAnswered] = useState(false);
  const scheduled = Object.keys(progress.review).length;
  const q = ready && queue[pos] ? questions![queue[pos]] : undefined;

  return (
    <article className="page" data-ready={ready ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>{t('Review', 'Ôn tập')}</h1>
        <p className="lede">
          {t(
            `Every question you answer in a lesson comes back here after ${INTERVAL_DAYS.join(', ')} days, moving up a step each time you get it right and back to the start when you miss.`,
            `Mỗi câu bạn trả lời trong bài học sẽ xuất hiện lại ở đây sau ${INTERVAL_DAYS.join(', ')} ngày. Trả lời đúng sẽ giãn khoảng cách đến lần ôn tiếp theo thêm một bậc. Trả lời sai đưa câu hỏi về bậc đầu.`,
          )}
        </p>
      </header>
      <div className="prose">
        {!ready ? (
          <p className="muted">{t('Loading…', 'Đang tải…')}</p>
        ) : q ? (
          <>
            <p className="status-line">
              {t(`Question ${pos + 1} of ${queue.length} due today`, `Câu ${pos + 1}/${queue.length} đến hạn hôm nay`)}
            </p>
            <Ask key={q.id} q={q} review label={t('From an earlier lesson', 'Từ một bài đã học')} onResult={() => setAnswered(true)} />
            <p>
              <button className="primary" disabled={!answered} onClick={() => { setPos(pos + 1); setAnswered(false); }}>
                {pos + 1 < queue.length ? t('Next question', 'Câu tiếp theo') : t('Finish', 'Kết thúc')}
              </button>
            </p>
          </>
        ) : (
          <p>
            {queue.length
              ? t('That was the last one due today.', 'Bạn đã ôn hết các câu đến hạn hôm nay.')
              : scheduled
                ? t(`Nothing is due today. ${scheduled} question${scheduled === 1 ? ' is' : 's are'} scheduled for later.`, `Hôm nay chưa có câu nào đến hạn. Có ${scheduled} câu đã được lên lịch ôn sau.`)
                : t('Nothing is scheduled yet. Questions enter the schedule the first time you answer them in a lesson.', 'Chưa có câu nào trong lịch ôn. Mỗi câu được đưa vào lịch sau lần đầu bạn trả lời trong bài học.')}
          </p>
        )}
      </div>
    </article>
  );
}
