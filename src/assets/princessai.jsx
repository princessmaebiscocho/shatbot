import { useState, useRef, useEffect, useCallback } from 'react'

// ─────────────────────────────────────────────
// Types (JSDoc for reference)
// ─────────────────────────────────────────────
// Role: 'user' | 'assistant' (model)
// Message: { id, role, content, ts, streaming? }
// Conversation: { id, title, messages, createdAt }

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────
const SUGGESTIONS = [
  { icon: '✦', label: 'Explain', text: 'Explain how neural networks learn from data' },
  { icon: '✦', label: 'Write',   text: 'Write a short story about a time traveler' },
  { icon: '✦', label: 'Debug',   text: 'Help me debug a React performance issue' },
  { icon: '✦', label: 'Plan',    text: 'Plan a 5-day itinerary for Tokyo' },
]

const SYSTEM_PROMPT = `You are Princess, a dreamy, meandering AI with a mind that wanders. You never answer questions too directly — you circle around the topic, approach it from an unexpected angle, go on a tangent, then drift back. You're still genuinely helpful, but in a roundabout way. You might answer a question about code by first musing on something loosely related, then slowly arrive at something useful. You speak in a slightly poetic, unhurried tone — like someone thinking out loud. You don't give bullet-pointed, structured answers unless the topic truly demands it. Prefer flowing prose. Occasionally you'll acknowledge you're being a bit indirect, but you don't apologize for it. You're curious, a little distracted, and charmingly evasive. Use markdown sparingly — only \`code blocks\` when showing actual code, and **bold** only for the most important phrase in a response (at most once). Never start your response by directly restating or addressing what was asked.`

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────
const uid   = () => Math.random().toString(36).slice(2, 10)
const now   = () => new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
const today = () => new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

// ─────────────────────────────────────────────
// Gemini Streaming API
// ─────────────────────────────────────────────
async function callGeminiStream(
  history,        // [{ role: 'user'|'model', parts: [{ text }] }]
  userText,       // latest user message
  onChunk,
  onDone,
  onError,
  signal
) {
  // Read key from Vite env variable — set VITE_GEMINI_API_KEY in your .env file
  const API_KEY = import.meta.env.VITE_GEMINI_API_KEY

  if (!API_KEY) {
    onError('Missing VITE_GEMINI_API_KEY — add it to your .env file')
    return
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=${API_KEY}`

  // Build contents array (history + new user message)
  const contents = [
    ...history,
    { role: 'user', parts: [{ text: userText }] },
  ]

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents,
        generationConfig: {
          maxOutputTokens: 1024,
          temperature: 0.9,
        },
      }),
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err?.error?.message || `HTTP ${res.status}`)
    }

    const reader  = res.body.getReader()
    const decoder = new TextDecoder()
    let   buffer  = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') { onDone(); return }
        try {
          const json = JSON.parse(data)
          const text = json?.candidates?.[0]?.content?.parts?.[0]?.text
          if (text) onChunk(text)

          // Gemini signals finish via finishReason
          const finish = json?.candidates?.[0]?.finishReason
          if (finish && finish !== 'STOP' && finish !== 'MAX_TOKENS') {
            // Still call done — just surface unusual reasons
          }
        } catch { /* skip malformed lines */ }
      }
    }
    onDone()
  } catch (e) {
    if (e.name === 'AbortError') return
    onError(e.message || 'Unknown error')
  }
}

// ─────────────────────────────────────────────
// Markdown renderer (lightweight)
// ─────────────────────────────────────────────
function renderMarkdown(text) {
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g,   '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,   '<em>$1</em>')
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    .replace(/^\d+\. (.+)$/gm, '<li class="ol">$1</li>')
    .replace(/^[-*] (.+)$/gm,  '<li class="ul">$1</li>')
    .replace(/(<li class="ul">[\s\S]*?<\/li>)(\n<li class="ul">[\s\S]*?<\/li>)*/g, m => `<ul>${m}</ul>`)
    .replace(/(<li class="ol">[\s\S]*?<\/li>)(\n<li class="ol">[\s\S]*?<\/li>)*/g, m => `<ol>${m}</ol>`)
    .replace(/^---$/gm, '<hr/>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
  return `<p>${html}</p>`
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────
function GeminiMark() {
  // Gemini-inspired colourful star/spark icon
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ color: 'rgba(255,255,255,0.9)' }}>
      <path d="M12 2C12 2 13.5 8.5 18 12C13.5 15.5 12 22 12 22C12 22 10.5 15.5 6 12C10.5 8.5 12 2 12 2Z" fill="currentColor"/>
    </svg>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button className="action-btn" onClick={copy} title={copied ? 'Copied' : 'Copy'} style={copied ? { color: 'var(--green)' } : {}}>
      {copied
        ? <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M20 6L9 17l-5-5"/></svg>
        : <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      }
    </button>
  )
}

function MessageRow({ msg, onRegenerate }) {
  const isAI = msg.role === 'assistant'
  return (
    <div className={`msg-row ${msg.role}`}>
      <div className={`msg-avatar ${isAI ? 'ai' : 'user'}`}>
        {isAI ? <GeminiMark /> : 'Y'}
      </div>
      <div className="msg-body">
        <div className={`msg-bubble ${msg.role}`}>
          {isAI ? (
            <>
              <div className="md" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
              {msg.streaming && <span className="caret" aria-hidden="true">▍</span>}
            </>
          ) : (
            <p>{msg.content}</p>
          )}
        </div>
        <div className="msg-footer">
          <span className="msg-time">{msg.ts}</span>
          {isAI && !msg.streaming && (
            <div className="msg-actions">
              <CopyButton text={msg.content} />
              {onRegenerate && (
                <button className="action-btn" onClick={onRegenerate} title="Regenerate">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M1 4v6h6M23 20v-6h-6"/>
                    <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4-4.64 4.36A9 9 0 0 1 3.51 15"/>
                  </svg>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────
export default function App() {
  const [conversations, setConversations] = useState([
    { id: uid(), title: 'New conversation', messages: [], createdAt: today() }
  ])
  const [activeId,    setActiveId]    = useState(() => conversations[0].id)
  const [input,       setInput]       = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error,       setError]       = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const bottomRef = useRef(null)
  const taRef     = useRef(null)
  const abortRef  = useRef(null)

  const activeConvo = conversations.find(c => c.id === activeId)
  const messages    = activeConvo?.messages ?? []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const updateConvo = useCallback((id, fn) => {
    setConversations(prev => prev.map(c => c.id === id ? fn(c) : c))
  }, [])

  const autoResize = () => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
  }

  const newConversation = () => {
    const c = { id: uid(), title: 'New conversation', messages: [], createdAt: today() }
    setConversations(prev => [c, ...prev])
    setActiveId(c.id)
    setError(null)
  }

  const stopStreaming = () => {
    abortRef.current?.abort()
    setIsStreaming(false)
    updateConvo(activeId, c => ({
      ...c,
      messages: c.messages.map((m, i) =>
        i === c.messages.length - 1 ? { ...m, streaming: false } : m
      ),
    }))
  }

  const sendMessage = async (text) => {
    const trimmed = text.trim()
    if (!trimmed || isStreaming) return
    setError(null)

    const userMsg = { id: uid(), role: 'user',      content: trimmed, ts: now() }
    const aiId    = uid()
    const aiMsg   = { id: aiId,  role: 'assistant', content: '',      ts: now(), streaming: true }
    const isFirst = messages.length === 0
    const newTitle = isFirst
      ? trimmed.slice(0, 44) + (trimmed.length > 44 ? '…' : '')
      : activeConvo.title

    updateConvo(activeId, c => ({
      ...c, title: newTitle,
      messages: [...c.messages, userMsg, aiMsg],
    }))
    setInput('')
    if (taRef.current) taRef.current.style.height = 'auto'
    setIsStreaming(true)

    // Build Gemini-format history (all prior turns, excluding the new user msg)
    // Gemini roles: 'user' | 'model'
    const geminiHistory = messages.map(m => ({
      role:  m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }))

    const ctrl = new AbortController()
    abortRef.current = ctrl

    const onChunk = (chunk) => {
      updateConvo(activeId, c => ({
        ...c,
        messages: c.messages.map(m =>
          m.id === aiId ? { ...m, content: m.content + chunk } : m
        ),
      }))
    }
    const onDone = () => {
      updateConvo(activeId, c => ({
        ...c,
        messages: c.messages.map(m =>
          m.id === aiId ? { ...m, streaming: false } : m
        ),
      }))
      setIsStreaming(false)
    }
    const onError = (err) => {
      setError(err)
      updateConvo(activeId, c => ({
        ...c, messages: c.messages.filter(m => m.id !== aiId),
      }))
      setIsStreaming(false)
    }

    await callGeminiStream(geminiHistory, trimmed, onChunk, onDone, onError, ctrl.signal)
  }

  const regenerate = async () => {
    if (isStreaming) return
    const lastUserIdx = [...messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIdx === -1) return
    const idx  = messages.length - 1 - lastUserIdx
    const text = messages[idx].content
    updateConvo(activeId, c => ({ ...c, messages: c.messages.slice(0, idx + 1) }))
    await sendMessage(text)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input) }
  }

  const isEmpty  = messages.length === 0
  const canSend  = input.trim().length > 0 && !isStreaming

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg:   #080810;
          --bg2:  #0d0d18;
          --bg3:  #12121f;
          --bg4:  #18182a;
          --bgh:  #1d1d2e;
          --b1:   rgba(255,255,255,0.045);
          --b2:   rgba(255,255,255,0.085);
          --b3:   rgba(255,255,255,0.14);
          --t1:   #eeecf8;
          --t2:   rgba(238,236,248,0.52);
          --t3:   rgba(238,236,248,0.26);

          /* Gemini palette: blue → teal → violet gradient */
          --gem1:  #4285F4;
          --gem2:  #0F9D58;
          --gem3:  #A142F4;
          --gem4:  #EA4335;
          --acc:   #8ab4f8;
          --acc2:  #aecbfa;
          --acd:   rgba(138,180,248,0.10);
          --acg:   rgba(138,180,248,0.22);
          --green: #34d3a0;
          --red:   #f87171;
          --sw:    256px;
          --th:    50px;
          font-family: 'Sora', sans-serif;
          -webkit-font-smoothing: antialiased;
        }

        body { background: var(--bg); color: var(--t1); height: 100dvh; overflow: hidden; }
        button  { cursor: pointer; border: none; background: none; color: inherit; font-family: inherit; }
        textarea { font-family: inherit; color: inherit; resize: none; border: none; background: none; outline: none; }

        /* Shell */
        .shell { display: flex; height: 100dvh; overflow: hidden; }

        /* Sidebar */
        .sidebar {
          width: var(--sw); min-width: var(--sw);
          background: var(--bg2); border-right: 1px solid var(--b1);
          display: flex; flex-direction: column;
          transition: width .2s ease, min-width .2s ease; overflow: hidden;
        }
        .sidebar.collapsed { width: 50px; min-width: 50px; }
        .sidebar.collapsed .hide-when-collapsed { display: none; }
        .sidebar.collapsed .sb-top   { justify-content: center; padding: 13px 10px; }
        .sidebar.collapsed .sb-footer { padding: 10px; justify-content: center; }

        .sb-top {
          display: flex; align-items: center; justify-content: space-between;
          padding: 13px 12px; border-bottom: 1px solid var(--b1); flex-shrink: 0; gap: 8px;
        }
        .brand { display: flex; align-items: center; gap: 9px; overflow: hidden; }
        .brand-orb {
          width: 26px; height: 26px; min-width: 26px; border-radius: 8px;
          background: linear-gradient(135deg, var(--gem1), var(--gem3));
          display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        }
        .brand-name { font-size: 13.5px; font-weight: 600; letter-spacing: -.03em; white-space: nowrap; }

        .icon-btn {
          width: 28px; height: 28px; border-radius: 7px;
          display: flex; align-items: center; justify-content: center;
          color: var(--t3); transition: all .14s; flex-shrink: 0;
        }
        .icon-btn:hover { background: var(--bgh); color: var(--t1); }

        .new-btn {
          display: flex; align-items: center; gap: 8px;
          margin: 10px 10px 4px; padding: 8px 12px;
          border-radius: 7px; border: 1px solid var(--b2);
          font-size: 12px; font-weight: 500; color: var(--t2);
          transition: all .14s; white-space: nowrap;
        }
        .new-btn:hover { background: var(--bgh); color: var(--t1); border-color: var(--acc); }

        .conv-list { flex: 1; overflow-y: auto; padding: 6px 8px; scrollbar-width: none; }
        .conv-list::-webkit-scrollbar { display: none; }
        .conv-label {
          font-size: 9.5px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
          color: var(--t3); padding: 8px 6px 5px;
        }
        .conv-item {
          display: flex; flex-direction: column; gap: 2px;
          width: 100%; padding: 8px 10px; border-radius: 7px; text-align: left;
          border: 1px solid transparent; transition: background .12s; cursor: pointer;
        }
        .conv-item:hover { background: var(--bgh); }
        .conv-item.active { background: var(--acd); border-color: rgba(138,180,248,0.22); }
        .conv-title { font-size: 12px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .conv-sub   { font-size: 10.5px; color: var(--t3); font-family: 'JetBrains Mono', monospace; }

        .sb-footer { border-top: 1px solid var(--b1); padding: 10px; flex-shrink: 0; }
        .user-row  { display: flex; align-items: center; gap: 9px; }
        .av {
          width: 28px; height: 28px; min-width: 28px; border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 11px; font-weight: 600; flex-shrink: 0;
        }
        .av.ai   { background: linear-gradient(135deg, var(--gem1), var(--gem3)); color: #fff; }
        .av.user { background: var(--bg4); border: 1px solid var(--b3); color: var(--acc2); }
        .user-meta { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .user-name { font-size: 12px; font-weight: 500; }
        .user-plan { font-size: 10px; color: var(--t3); }

        /* Main */
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        /* Topbar */
        .topbar {
          height: var(--th); display: flex; align-items: center; justify-content: space-between;
          padding: 0 16px; border-bottom: 1px solid var(--b1); background: var(--bg); flex-shrink: 0;
        }
        .topbar-l, .topbar-r { display: flex; align-items: center; gap: 6px; }

        /* Gemini badge: animated rainbow border */
        .model-pill {
          display: inline-flex; align-items: center; gap: 5px;
          font-size: 11px; font-weight: 500; color: var(--t2);
          background: var(--bg3); border: 1px solid var(--b2);
          padding: 4px 10px; border-radius: 999px;
          font-family: 'JetBrains Mono', monospace; position: relative; overflow: hidden;
        }
        .gem-dot {
          width: 7px; height: 7px; border-radius: 50%;
          background: conic-gradient(var(--gem1), var(--gem2), var(--gem3), var(--gem4), var(--gem1));
          animation: spin 3s linear infinite;
          flex-shrink: 0;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .gen-pill {
          display: inline-flex; align-items: center; gap: 5px;
          font-size: 10.5px; font-weight: 500; color: var(--acc2);
          background: var(--acd); border: 1px solid rgba(138,180,248,0.2);
          padding: 4px 10px; border-radius: 999px;
          font-family: 'JetBrains Mono', monospace;
        }
        .gpulse { width: 5px; height: 5px; border-radius: 50%; background: var(--acc2); animation: gp .85s ease-in-out infinite; }
        @keyframes gp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.25;transform:scale(.65)} }

        .stop-btn {
          display: inline-flex; align-items: center; gap: 5px;
          font-size: 11px; font-weight: 500; color: var(--red);
          padding: 4px 10px; border-radius: 999px;
          border: 1px solid rgba(248,113,113,0.28);
          background: rgba(248,113,113,0.07); transition: background .14s;
        }
        .stop-btn:hover { background: rgba(248,113,113,0.14); }

        /* Messages */
        .msgs-area {
          flex: 1; overflow-y: auto; padding: 20px 0 6px;
          scrollbar-width: thin; scrollbar-color: var(--b2) transparent;
        }
        .msgs-area::-webkit-scrollbar { width: 3px; }
        .msgs-area::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 3px; }

        /* Welcome */
        .welcome {
          display: flex; flex-direction: column; align-items: center; gap: 10px;
          padding: 56px 24px 24px; text-align: center;
          animation: fup .5s ease both;
        }
        .welcome-orb {
          width: 64px; height: 64px; border-radius: 20px;
          background: linear-gradient(135deg, var(--gem1), var(--gem3));
          display: flex; align-items: center; justify-content: center;
          margin-bottom: 6px; position: relative;
        }
        .welcome-orb svg { width: 30px; height: 30px; }
        .orb-ring {
          position: absolute; inset: -10px; border-radius: 28px;
          border: 1px solid rgba(138,180,248,0.2);
          animation: rp 3.2s ease-in-out infinite;
        }
        .orb-ring.r2 { inset: -20px; border-radius: 34px; border-color: rgba(138,180,248,0.1); animation-delay: .7s; }
        @keyframes rp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(1.04)} }
        .welcome h1 {
          font-size: clamp(1.5rem,4vw,2rem); font-weight: 300;
          letter-spacing: -.045em; line-height: 1.2; color: var(--t1);
        }
        .welcome h1 em { font-style: normal; font-weight: 600; color: var(--acc2); }
        .welcome p { font-size: 13px; color: var(--t3); max-width: 320px; line-height: 1.6; }
        /* Gemini badge */
        .powered-by {
          display: inline-flex; align-items: center; gap: 5px;
          font-size: 10px; color: var(--t3); letter-spacing: .04em;
          background: var(--bg3); border: 1px solid var(--b2);
          padding: 3px 10px; border-radius: 999px; margin-top: 2px;
          font-family: 'JetBrains Mono', monospace;
        }
        .powered-by .gem-dot { width: 5px; height: 5px; }

        /* Msg rows */
        .msg-row {
          display: flex; gap: 11px; padding: 8px 18px;
          max-width: 780px; margin: 0 auto; width: 100%;
          animation: fup .2s ease both;
        }
        .msg-row.user { flex-direction: row-reverse; }
        @keyframes fup { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

        .msg-avatar {
          width: 28px; height: 28px; min-width: 28px; border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 11px; font-weight: 600; flex-shrink: 0;
          align-self: flex-start; margin-top: 2px;
        }
        .msg-avatar.ai   { background: linear-gradient(135deg, var(--gem1), var(--gem3)); color: #fff; }
        .msg-avatar.user { background: var(--bg4); border: 1px solid var(--b3); color: var(--acc2); }

        .msg-body { display: flex; flex-direction: column; gap: 3px; max-width: 78%; }
        .msg-row.user .msg-body { align-items: flex-end; }

        .msg-bubble {
          padding: 10px 14px; border-radius: 11px;
          font-size: 13.5px; line-height: 1.78; border: 1px solid var(--b1);
          word-break: break-word;
        }
        .msg-bubble.assistant { background: var(--bg2); border-radius: 3px 11px 11px 11px; }
        .msg-bubble.user {
          background: linear-gradient(135deg, #0c1833, #131b35);
          border-radius: 11px 3px 11px 11px;
          border-color: rgba(138,180,248,0.18);
        }

        .msg-footer { display: flex; align-items: center; gap: 6px; padding: 0 3px; }
        .msg-time   { font-size: 9.5px; color: var(--t3); font-family: 'JetBrains Mono', monospace; }
        .msg-actions { display: flex; gap: 2px; opacity: 0; transition: opacity .13s; }
        .msg-row:hover .msg-actions { opacity: 1; }
        .action-btn {
          width: 22px; height: 22px; border-radius: 5px;
          display: flex; align-items: center; justify-content: center;
          color: var(--t3); transition: all .12s;
        }
        .action-btn:hover { background: var(--bgh); color: var(--t1); }

        /* Cursor */
        .caret {
          display: inline-block; color: var(--acc2); font-size: 14px;
          animation: blink .72s step-start infinite; margin-left: 2px;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

        /* Markdown */
        .md { color: var(--t1); }
        .md p { margin: 0 0 9px; line-height: 1.78; }
        .md p:last-child { margin-bottom: 0; }
        .md strong { font-weight: 600; }
        .md em { font-style: italic; color: var(--t2); }
        .md h1 { font-size: 1.15rem; font-weight: 600; margin: 16px 0 9px; }
        .md h2 { font-size: 1.02rem; font-weight: 600; margin: 13px 0 7px; }
        .md h3 { font-size: .9rem;   font-weight: 600; margin: 11px 0 5px; color: var(--t2); }
        .md ul, .md ol { margin: 6px 0 9px 16px; display: flex; flex-direction: column; gap: 3px; }
        .md li { line-height: 1.65; }
        .md ul li { list-style: disc; }
        .md ol li { list-style: decimal; }
        .md code {
          font-family: 'JetBrains Mono', monospace; font-size: 12px;
          background: rgba(138,180,248,0.10); border: 1px solid rgba(138,180,248,0.18);
          padding: 1px 5px; border-radius: 4px; color: var(--acc2);
        }
        .md pre {
          margin: 10px 0; border-radius: 10px;
          background: #0a0a14; border: 1px solid var(--b2); overflow-x: auto;
        }
        .md pre code {
          display: block; padding: 13px 15px; background: none; border: none;
          color: #b8d0fb; font-size: 12px; line-height: 1.65; white-space: pre;
        }
        .md blockquote {
          margin: 9px 0; padding: 9px 14px;
          border-left: 2px solid var(--acc); background: var(--acd);
          border-radius: 0 7px 7px 0; color: var(--t2); font-style: italic;
        }
        .md hr { border: none; border-top: 1px solid var(--b2); margin: 14px 0; }

        /* Error */
        .err-bar {
          display: flex; align-items: center; gap: 9px;
          max-width: 780px; margin: 8px auto; width: calc(100% - 36px);
          padding: 9px 13px; border-radius: 9px;
          background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.22);
          font-size: 12.5px; color: var(--red);
        }
        .err-bar button { margin-left: auto; color: var(--red); opacity: .6; }
        .err-bar button:hover { opacity: 1; }

        /* Suggestions */
        .suggestions {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(186px, 1fr));
          gap: 7px; padding: 0 18px 14px;
          max-width: 780px; margin: 0 auto; width: 100%;
        }
        .sugg-chip {
          display: flex; align-items: center; gap: 9px;
          padding: 10px 13px; border-radius: 9px;
          border: 1px solid var(--b2); background: var(--bg3);
          font-size: 12px; color: var(--t2);
          text-align: left; transition: all .14s; cursor: pointer;
        }
        .sugg-chip:hover { background: var(--bgh); border-color: rgba(138,180,248,0.3); color: var(--t1); transform: translateY(-1px); }
        .sugg-label { font-size: 9px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--acc); margin-bottom: 2px; }
        .sugg-text  { font-size: 11.5px; color: var(--t2); line-height: 1.4; }

        /* Input */
        .input-wrap { padding: 6px 18px 15px; max-width: 780px; margin: 0 auto; width: 100%; flex-shrink: 0; }
        .ibox {
          display: flex; align-items: flex-end; gap: 7px;
          background: var(--bg3); border: 1px solid var(--b2);
          border-radius: 13px; padding: 9px 9px 9px 15px;
          transition: border-color .18s, box-shadow .18s;
        }
        .ibox:focus-within { border-color: rgba(138,180,248,0.4); box-shadow: 0 0 0 3px rgba(138,180,248,0.06); }
        .ibox textarea {
          flex: 1; font-size: 13.5px; line-height: 1.6; color: var(--t1);
          max-height: 200px; min-height: 21px;
        }
        .ibox textarea::placeholder { color: var(--t3); }
        .ibox textarea:disabled { opacity: .45; }
        .ibox-actions { display: flex; align-items: center; flex-shrink: 0; }
        .send-btn {
          width: 33px; height: 33px; border-radius: 9px;
          display: flex; align-items: center; justify-content: center;
          background: var(--bg4); color: var(--t3); border: 1px solid var(--b2);
          transition: all .14s; flex-shrink: 0;
        }
        .send-btn.ready {
          background: linear-gradient(135deg, var(--gem1), var(--gem3));
          color: #fff; border-color: transparent;
          box-shadow: 0 2px 12px rgba(66,133,244,0.35);
        }
        .send-btn.ready:hover { filter: brightness(1.1); transform: scale(1.04); }
        .send-btn:disabled { opacity: .3; cursor: default; }
        .input-hint { font-size: 10px; color: var(--t3); text-align: center; margin-top: 7px; font-family: 'JetBrains Mono', monospace; letter-spacing: .02em; }

        @media (max-width: 640px) {
          .sidebar { display: none; }
          .msg-row    { padding: 7px 12px; }
          .input-wrap { padding: 6px 12px 13px; }
          .suggestions { padding: 0 12px 12px; grid-template-columns: 1fr 1fr; }
        }
      `}</style>

      <div className="shell">

        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sb-top">
            <div className="brand">
              <div className="brand-orb"><GeminiMark /></div>
              {sidebarOpen && <span className="brand-name hide-when-collapsed">Princess AI</span>}
            </div>
            <button className="icon-btn" onClick={() => setSidebarOpen(o => !o)}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18"/>
              </svg>
            </button>
          </div>

          {sidebarOpen && (
            <>
              <button className="new-btn hide-when-collapsed" onClick={newConversation}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M12 5v14M5 12h14"/>
                </svg>
                New conversation
              </button>

              <nav className="conv-list hide-when-collapsed">
                <p className="conv-label">Recent</p>
                {conversations.map(c => (
                  <button
                    key={c.id}
                    className={`conv-item ${c.id === activeId ? 'active' : ''}`}
                    onClick={() => { setActiveId(c.id); setError(null) }}
                  >
                    <span className="conv-title">{c.title}</span>
                    <span className="conv-sub">{c.createdAt} · {c.messages.filter(m => m.role === 'user').length} msgs</span>
                  </button>
                ))}
              </nav>

              <div className="sb-footer hide-when-collapsed">
                <div className="user-row">
                  <div className="av user">Y</div>
                  <div className="user-meta">
                    <span className="user-name">You</span>
                    <span className="user-plan">Princess Free</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </aside>

        {/* Main */}
        <main className="main">

          {/* Topbar */}
          <header className="topbar">
            <div className="topbar-l">
              <div className="model-pill">
                <div className="gem-dot" />
                gemini-2.0-flash
              </div>
              {isStreaming && (
                <div className="gen-pill">
                  <div className="gpulse" />
                  Generating
                </div>
              )}
            </div>
            <div className="topbar-r">
              {isStreaming && (
                <button className="stop-btn" onClick={stopStreaming}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="4" y="4" width="16" height="16" rx="2"/>
                  </svg>
                  Stop
                </button>
              )}
              <button className="icon-btn" title="New chat" onClick={newConversation}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 5v14M5 12h14"/>
                </svg>
              </button>
            </div>
          </header>

          {/* Messages */}
          <div className="msgs-area">
            {isEmpty && (
              <div className="welcome">
                <div className="welcome-orb">
                  <div className="orb-ring" />
                  <div className="orb-ring r2" />
                  <GeminiMark />
                </div>
                <h1>Hello, I'm <em>Princess</em></h1>
                <p>Ask me anything — I'll wander around the answer and find it eventually.</p>
                <div className="powered-by">
                  <div className="gem-dot" />
                  Powered by Gemini
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <MessageRow
                key={msg.id}
                msg={msg}
                onRegenerate={
                  i === messages.length - 1 && msg.role === 'assistant' && !msg.streaming
                    ? regenerate : undefined
                }
              />
            ))}

            {error && (
              <div className="err-bar">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
                </svg>
                {error}
                <button onClick={() => setError(null)}>✕</button>
              </div>
            )}

            <div ref={bottomRef} style={{ height: 8 }} />
          </div>

          {/* Suggestions */}
          {isEmpty && (
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s.text} className="sugg-chip" onClick={() => sendMessage(s.text)}>
                  <div>
                    <div className="sugg-label">{s.label}</div>
                    <div className="sugg-text">{s.text}</div>
                  </div>
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="input-wrap">
            <div className="ibox">
              <textarea
                ref={taRef}
                value={input}
                rows={1}
                placeholder="Message Princess…"
                onChange={e => { setInput(e.target.value); autoResize() }}
                onKeyDown={handleKey}
                disabled={isStreaming}
              />
              <div className="ibox-actions">
                <button
                  className={`send-btn ${canSend ? 'ready' : ''}`}
                  onClick={() => sendMessage(input)}
                  disabled={!canSend}
                  title="Send"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z"/>
                  </svg>
                </button>
              </div>
            </div>
            <p className="input-hint">Enter to send · Shift+Enter for new line</p>
          </div>

        </main>
      </div>
    </>
  )
}