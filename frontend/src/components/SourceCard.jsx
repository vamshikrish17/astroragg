function SourceCard({ source }) {
  const score =
    typeof source?.score === 'number' ? `${Math.round(source.score * 100)}%` : 'N/A'

  return (
    <article className="source-card">
      <div>
        <span className="source-label">Mission File</span>
        <h3>{source?.document || 'Unknown source'}</h3>
      </div>
      <span className="source-score">{score}</span>
    </article>
  )
}

export default SourceCard
