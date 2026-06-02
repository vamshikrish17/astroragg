import ChatWindow from './components/ChatWindow'
import Header from './components/Header'
import './styles/global.css'
import './styles/space-theme.css'
import './styles/chat.css'

function App() {
  return (
    <main className="app-shell">
      <div className="space-background" aria-hidden="true">
        <div className="star-field star-field-a"></div>
        <div className="star-field star-field-b"></div>
        <div className="nebula nebula-cyan"></div>
        <div className="nebula nebula-rose"></div>
      </div>

      <section className="research-console">
        <Header />
        <ChatWindow />
      </section>
    </main>
  )
}

export default App
