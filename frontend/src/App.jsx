import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const API = '/api'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('cs_session'))
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API}/chat/graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer, sources: data.sources }])
      if (data.session_id) {
        setSessionId(data.session_id)
        localStorage.setItem('cs_session', data.session_id)
      }
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: '网络错误，请稍后重试' }])
    } finally {
      setLoading(false)
    }
  }

  const clearChat = async () => {
    if (sessionId) {
      await fetch(`${API}/session/${sessionId}`, { method: 'DELETE' })
    }
    setMessages([])
    setSessionId(null)
    localStorage.removeItem('cs_session')
  }

  return (
    <div className="app">
      <div className="header">
            <h1>快修到家 AI 客服</h1>
            <span className="subtitle">家居安装服务 · 客户服务中心</span>
        <button className="btn-clear" onClick={clearChat}>
          新对话
        </button>
      </div>

      <div className="chat-container">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>您好，我是小快</h2>
            <p>请问有什么可以帮您？</p>
            <div className="quick-asks">
              {['空调安装费多少？', '师傅迟到了怎么办？', '怎么联系客服？'].map((q) => (
                <button key={q} className="btn-quick" onClick={() => { setInput(q); sendMessage() }}>
                  {q}
                </button>
              ))}
              <span style={{ display: 'none' }}>{input}</span>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="avatar">{msg.role === 'user' ? '我' : '快'}</div>
            <div className="bubble">
              <div className="text">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <details className="sources">
                  <summary>参考来源 ({msg.sources.length})</summary>
                  {msg.sources.map((s, j) => (
                    <div key={j} className="source-card">
                      <strong>{s.title || '知识库'}</strong>
                      <p>{s.content}</p>
                    </div>
                  ))}
                </details>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="avatar">快</div>
            <div className="bubble typing">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="输入您的问题..."
          disabled={loading}
          autoFocus
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          发送
        </button>
      </div>
    </div>
  )
}

export default App
