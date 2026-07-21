import type { Citation, SourceType } from "../lib/types";

/* ------------------------------------------------------------------
   The signature element.

   Citations split into two registers, because that split IS the
   product: code answers "what", the discussion around it answers
   "why". Code renders neutral, discussion renders in brass. When
   both are present the answer was assembled across the join between
   them, and a connecting rail is drawn to say so. When only one is
   present there is no rail. The absence is honest, not decorative.
   ------------------------------------------------------------------ */

const WHY_TYPES: SourceType[] = [
  "pr",
  "commit",
  "issue",
  "comment",
  "discussion",
];

/** Code chunks embed "File: path (lines 9-21)" as their first line. */
function parseLines(excerpt: string): string | null {
  const m = excerpt.match(/^File:\s+.*?\(lines\s+(\d+)-(\d+)\)/);
  return m ? `${m[1]}\u2013${m[2]}` : null;
}

/** Strips the location header so the excerpt shows actual content. */
function stripHeader(excerpt: string): string {
  return excerpt.replace(/^File:\s+.*?\(lines\s+\d+-\d+\)\n?/, "").trim();
}

function label(c: Citation): string {
  switch (c.source_type) {
    case "code":
      return c.source_id;
    case "commit":
      return `commit ${c.source_id.slice(0, 7)}`;
    case "pr":
      return `pull request #${c.source_id}`;
    case "issue":
      return `issue #${c.source_id}`;
    case "comment":
      return `comment on #${c.source_id}`;
    case "release":
      return `release ${c.source_id}`;
    default:
      return c.source_id;
  }
}

function Row({ c, why }: { c: Citation; why: boolean }) {
  const lines = c.source_type === "code" ? parseLines(c.excerpt) : null;
  const body =
    c.source_type === "code" ? stripHeader(c.excerpt) : c.excerpt.trim();

  const inner = (
    <>
      <div className="cite-top">
        <span className="cite-id">{label(c)}</span>
        {lines && <span className="cite-lines">{lines}</span>}
        {c.url && <span className="cite-arrow">&#8599;</span>}
      </div>
      {body && <div className="cite-excerpt">{body}</div>}
    </>
  );

  const cls = why ? "cite why" : "cite";

  // Chunks stored before URLs were captured have an empty url.
  // Render those as plain blocks rather than dead links.
  return c.url ? (
    <a className={cls} href={c.url} target="_blank" rel="noreferrer">
      {inner}
    </a>
  ) : (
    <div className={cls}>{inner}</div>
  );
}

export default function Citations({ items }: { items: Citation[] }) {
  if (!items.length) return null;

  const what = items.filter((c) => !WHY_TYPES.includes(c.source_type));
  const why = items.filter((c) => WHY_TYPES.includes(c.source_type));
  const joined = what.length > 0 && why.length > 0;

  return (
    <div className="cites">
      {what.length > 0 && (
        <div className="cite-group">
          <div className="cite-head">
            <span className="eyebrow">What the code says</span>
            <span className="cite-head-line" />
          </div>
          {what.map((c) => (
            <Row key={c.chunk_id} c={c} why={false} />
          ))}
        </div>
      )}

      {joined && (
        <div className="join" aria-hidden="true">
          <span className="join-rail" />
          <span className="join-label">shaped by</span>
        </div>
      )}

      {why.length > 0 && (
        <div className="cite-group">
          <div className="cite-head">
            <span className="eyebrow">
              {joined ? "Why it's that way" : "From the history"}
            </span>
            <span className="cite-head-line" />
          </div>
          {why.map((c) => (
            <Row key={c.chunk_id} c={c} why />
          ))}
        </div>
      )}
    </div>
  );
}
