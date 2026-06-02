import SourceCard from './SourceCard'

function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message-row ${isUser ? 'message-row-user' : 'message-row-ai'}`}>
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'ai-bubble'}`}>
        <p>{message.content}</p>

        {!isUser && message.sources?.length > 0 && (
          <div className="source-grid">
            {message.sources.map((source) => (
              <SourceCard key={`${source.document}-${source.score}`} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageBubble
