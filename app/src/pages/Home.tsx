import { PLAN, useAllChapters } from '../chapters/registry';
import { href } from '../lib/router';
import { useProgress } from '../lib/progress';
import { dueCount } from '../lib/review';
import { useReports } from '../lib/reports';
import { useLanguage } from '../lib/language';
import { vietnamesePlan, vietnameseTier } from '../chapters/vietnamese';

export function Home() {
  const { progress } = useProgress();
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  const { reports } = useReports();
  const chapters = useAllChapters();
  const due = dueCount(progress.review);
  const backends = Object.entries(reports);
  const built = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  return (
    <article className="page" data-ready={chapters ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>{t('From tensors to kernels', 'Từ tensor đến kernel')}</h1>
        <p className="lede">
          {vi ? (
            <>Bản học và thực hành theo <i>Programming Massively Parallel Processors</i> (ấn bản 5), dành cho một
              người học. Bạn đã biết <code>torch.matmul</code> tính gì. Ở đây, bạn sẽ tìm hiểu GPU thực hiện phép tính
              đó ra sao và tự viết kernel bằng CUDA, HIP và Metal.</>
          ) : (
            <>A working copy of <i>Programming Massively Parallel Processors</i> (5th edition) for one reader: you know
              what <code>torch.matmul</code> computes, and this is about what the GPU does to compute it, and how to write
              that yourself on CUDA, HIP and Metal.</>
          )}
        </p>
      </header>
      <div className="prose">
        <h2 style={{ marginTop: 0 }}>{t('Where you are', 'Tiến độ học của bạn')}</h2>
        <ul>
          {PLAN.filter((r) => r.id && built.has(r.id)).map((r) => {
            const ch = built.get(r.id!)!;
            const done = ch.sections.filter((s) => progress.sections[`${ch.id}/${s.id}`]?.done).length;
            return (
              <li key={ch.id}>
                <a href={href(ch.id, ch.sections[0].id)}>
                  {/^\d/.test(ch.num) ? t('Chapter ', 'Chương ') : ch.num.startsWith('X') ? t('Module ', 'Mô-đun ') : t('Appendix ', 'Phụ lục ')}
                  {ch.num}: {vi ? vietnamesePlan[ch.id]?.title ?? ch.title : ch.title}
                </a>{' '}
                <span className="ui muted">
                  {t(`${done} of ${ch.sections.length} sections done`, `Đã hoàn thành ${done}/${ch.sections.length} mục`)}
                </span>
              </li>
            );
          })}
          <li>
            <a href={href('review')}>{t('Review', 'Ôn tập')}</a>{' '}
            <span className="ui muted">{due ? t(`${due} due today`, `${due} câu đến hạn hôm nay`) : t('nothing due today', 'hôm nay chưa có câu đến hạn')}</span>
          </li>
        </ul>

        <h2>{t('Your devices', 'Thiết bị của bạn')}</h2>
        {backends.length ? (
          <table className="data wide" style={{ maxWidth: '52rem' }}>
            <thead>
              <tr>
                <th>Backend</th>
                <th>{t('Device, as it reports itself', 'Tên do thiết bị báo cáo')}</th>
                <th>{t('Last run', 'Lần chạy gần nhất')}</th>
              </tr>
            </thead>
            <tbody>
              {backends.map(([name, run]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{String(run.device.name ?? t('unknown', 'không rõ'))}</td>
                  <td>{run.generated.replace('T', ' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>
            {vi ? (
              <>Máy này chưa có kết quả lab. Chạy <code>./verify.sh</code> ở thư mục gốc của dự án. Các bài có phần so
                sánh hiệu năng sẽ đọc số liệu đo từ máy của bạn.</>
            ) : (
              <>No lab results on this machine yet. Run <code>./verify.sh</code> in the repository root; lessons that compare
                against your own measurements will pick them up.</>
            )}
          </p>
        )}
        <p className="status-line">
          {t(
            'Nothing in the lessons assumes a particular GPU. Where the book works an example on a specific device, the lesson says so and puts your own measured numbers beside it.',
            'Bạn có thể học với bất kỳ GPU nào được hỗ trợ. Khi sách minh họa trên một thiết bị cụ thể, bài học sẽ nêu rõ thiết bị đó và đặt số liệu đo của bạn bên cạnh.',
          )}
        </p>

        <h2>{t('The whole course', 'Lộ trình toàn khóa')}</h2>
        <p>
          {t(
            'Depth follows your goals: matmul, attention, quantised operations and sampling for LLM inference, on more than one backend. Three modules are ours because the book does not cover them.',
            'Khóa học đi sâu vào nhân ma trận, attention, các phép toán lượng tử hóa và lấy mẫu cho suy luận LLM trên nhiều backend. Ba mô-đun bổ sung đề cập đến những chủ đề nằm ngoài sách.',
          )}
        </p>
        <table className="data wide">
          <thead>
            <tr>
              <th>{t('Ch.', 'Chương')}</th>
              <th>{t('Title', 'Chủ đề')}</th>
              <th>{t('Book pp.', 'Trang sách')}</th>
              <th>{t('Depth', 'Mức độ')}</th>
              <th>{t('Why', 'Nội dung trọng tâm')}</th>
              <th>{t('Project step', 'Mốc dự án')}</th>
              <th>{t('Status', 'Trạng thái')}</th>
            </tr>
          </thead>
          <tbody>
            {PLAN.map((r) => {
              const ch = r.id ? built.get(r.id) : undefined;
              const localized = vi ? vietnamesePlan[r.id ?? r.num] : undefined;
              const title = localized?.title ?? r.title;
              return (
                <tr key={r.num}>
                  <td>{r.num}</td>
                  <td>{ch ? <a href={href(ch.id, ch.sections[0].id)}>{title}</a> : title}</td>
                  <td className="nowrap">{vi && r.ours ? 'Ngoài sách' : r.pages}</td>
                  <td>{vi ? vietnameseTier[r.tier] : r.tier}</td>
                  <td>{localized?.why ?? r.why}</td>
                  <td>{localized?.milestone ?? r.milestone ?? ''}</td>
                  <td className="nowrap">{ch ? t('Ready', 'Sẵn sàng') : r.tier === 'Skipped' ? t('Not planned', 'Ngoài lộ trình') : t('Planned', 'Đã lên kế hoạch')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
