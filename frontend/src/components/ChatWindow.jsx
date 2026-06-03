import axios from 'axios'
import { useEffect, useRef, useState } from 'react'
import MessageBubble from './MessageBubble'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? '/_/backend' : 'http://127.0.0.1:8000')

const welcomeMessage = {
  id: crypto.randomUUID(),
  role: 'assistant',
  content:
    'Ask a question about the indexed space mission documents. I will answer from the retrieved PDF context and show the sources used.',
  sources: [],
}

function ChatWindow() {
  const [messages, setMessages] = useState([welcomeMessage])
  const [question, setQuestion] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = async (event) => {
    event.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isLoading) return

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: trimmedQuestion,
      sources: [],
    }

    setMessages((currentMessages) => [...currentMessages, userMessage])
    setQuestion('')
    setError('')
    setIsLoading(true)

    try {
      const response = await axios.post(`${API_BASE_URL}/ask`, {
        question: trimmedQuestion,
      })

      const aiMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.data.answer || 'No answer was returned by the backend.',
        sources: response.data.sources || [],
      }

      setMessages((currentMessages) => [...currentMessages, aiMessage])
    } catch (apiError) {
      const detail =
        apiError.response?.data?.detail ||
        apiError.message ||
        'The backend request failed.'
      setError(detail)
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `I could not complete the request. ${detail}`,
          sources: [],
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([welcomeMessage])
    setError('')
    setQuestion('')
  }

  return (
    <section className="chat-panel">
      <div className="chat-toolbar">
        <div>
          <span className="status-dot"></span>
          <span>Research link</span>
        </div>
        <button type="button" className="clear-button" onClick={clearChat}>
          Clear Chat
        </button>
      </div>

      <div className="chat-messages" role="log" aria-live="polite">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isLoading && (
          <div className="message-row message-row-ai">
            <div className="message-bubble ai-bubble loading-bubble">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={chatEndRef}></div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              handleSubmit(event)
            }
          }}
          placeholder="Ask about Chandrayaan-3, payloads, rovers, orbiters..."
          rows="1"
        />
        <button type="submit" disabled={isLoading || !question.trim()}>
          {isLoading ? 'Scanning' : 'Ask'}
        </button>
      </form>
    </section>
  )
}

export default ChatWindow
