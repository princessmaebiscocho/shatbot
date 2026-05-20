import { useState, useRef, useEffect, useCallback } from 'react'
import type { KeyboardEvent } from 'react'

type Role = 'user' | 'assistant'

interface Message {
  id: string
  role: Role
  content: string
  ts: string
  streaming?: boolean
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
}

const SUGGESTIONS = [
  { label: 'Explain', text: 'Explain how neural networks learn from data', icon: '◈' },
  { label: 'Write',   text: 'Write a short story about a time traveler', icon: '✦' },
  { label: 'Debug',   text: 'Help me debug a React performance issue', icon: '⌬' },
  { label: 'Plan',    text: 'Plan a 5-day itinerary for Tokyo', icon: '◉' },
]

const SYSTEM_PROMPT = `You are Princess, a dreamy, meandering AI with a mind that wanders. You never answer questions too directly — you circle around the topic, approach it from an unexpected angle, go on a tangent, then drift back. You're still genuinely helpful, but in a roundabout way. You might answer a question about code by first musing on something loosely related, then slowly arrive at something useful. You speak in a slightly poetic, unhurried tone — like someone thinking out loud. You don't give bullet-pointed, structured answers unless the topic truly demands it. Prefer flowing prose. Occasionally you'll acknowledge you're being a bit indirect, but you don't apologize for it. You're curious, a little distracted, and charmingly evasive. Use markdown sparingly — only \`code blocks\` when showing actual code, and **bold** only for the most important phrase in a response (at most once). Never start your response by directly restating or addressing what was asked.`

const uid   = () => Math.random().toString(36).slice(2, 10)
const now   = () => new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
const today = () => new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

type GeminiMessage = { role: 'user' | 'model'; parts: { text: string }[] }

async function callGeminiStream(
  history: GeminiMessage[],
  userText: string,
  onChunk: (chunk: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
  signal: AbortSignal
) {
  const API_KEY = import.meta.env.VITE_GEMINI_API_KEY as string | undefined
  if (!API_KEY) { onError('Missing VITE_GEMINI_API_KEY'); return }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=${API_KEY}`
  const contents: GeminiMessage[] = [...history, { role: 'user', parts: [{ text: userText }] }]

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents,
        generationConfig: { maxOutputTokens: 1024, temperature: 0.9 },
      }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error((err as { error?: { message?: string } }).error?.message || `HTTP ${res.status}`)
    }
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
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
        } catch { /* skip */ }
      }
    }
    onDone()
  } catch (e: unknown) {
    if ((e as { name?: string }).name === 'AbortError') return
    onError((e as Error).message || 'Unknown error')
  }
}

function renderMarkdown(text: string): string {
  const html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^\d+\. (.+)$/gm, '<li class="ol">$1</li>')
    .replace(/^[-*] (.+)$/gm, '<li class="ul">$1</li>')
    .replace(/(<li class="ul">[\s\S]*?<\/li>)(\n<li class="ul">[\s\S]*?<\/li>)*/g, m => `<ul>${m}</ul>`)
    .replace(/(<li class="ol">[\s\S]*?<\/li>)(\n<li class="ol">[\s\S]*?<\/li>)*/g, m => `<ol>${m}</ol>`)
    .replace(/^---$/gm, '<hr/>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
  return `<p>${html}</p>`
}

function StarMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C12 2 13.8 9 18 12C13.8 15 12 22 12 22C12 22 10.2 15 6 12C10.2 9 12 2 12 2Z" />
    </svg>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button className="action-btn" onClick={copy} title={copied ? 'Copied' : 'Copy'}>
      {copied
        ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M20 6L9 17l-5-5"/></svg>
        : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      }
    </button>
  )
}

function MessageRow({ msg, onRegenerate }: { msg: Message; onRegenerate?: () => void }) {
  const isAI = msg.role === 'assistant'
  return (
    <div className={`msg-row ${msg.role}`}>
      <div className={`msg-avatar ${isAI ? 'ai' : 'user'}`}>
        {isAI ? <StarMark /> : <span>Y</span>}
      </div>
      <div className="msg-body">
        <div className={`msg-bubble ${msg.role}`}>
          {isAI ? (
            <>
              <div className="md" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
              {msg.streaming && <span className="caret">▍</span>}
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
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([
    { id: uid(), title: 'New conversation', messages: [], createdAt: today() }
  ])
  const [activeId, setActiveId] = useState(() => conversations[0].id)
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const bottomRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const activeConvo = conversations.find(c => c.id === activeId)!
  const messages = activeConvo?.messages ?? []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const updateConvo = useCallback((id: string, fn: (c: Conversation) => Conversation) => {
    setConversations(prev => prev.map(c => c.id === id ? fn(c) : c))
  }, [])

  const autoResize = () => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px'
  }

  const newConversation = () => {
    const c: Conversation = { id: uid(), title: 'New conversation', messages: [], createdAt: today() }
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

  const sendMessage = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isStreaming) return
    setError(null)

    const userMsg: Message = { id: uid(), role: 'user', content: trimmed, ts: now() }
    const aiId = uid()
    const aiMsg: Message = { id: aiId, role: 'assistant', content: '', ts: now(), streaming: true }
    const isFirst = messages.length === 0
    const newTitle = isFirst ? trimmed.slice(0, 44) + (trimmed.length > 44 ? '…' : '') : activeConvo.title

    updateConvo(activeId, c => ({ ...c, title: newTitle, messages: [...c.messages, userMsg, aiMsg] }))
    setInput('')
    if (taRef.current) taRef.current.style.height = 'auto'
    setIsStreaming(true)

    const geminiHistory: GeminiMessage[] = messages.map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }))

    const ctrl = new AbortController()
    abortRef.current = ctrl

    await callGeminiStream(
      geminiHistory, trimmed,
      (chunk) => updateConvo(activeId, c => ({ ...c, messages: c.messages.map(m => m.id === aiId ? { ...m, content: m.content + chunk } : m) })),
      () => { updateConvo(activeId, c => ({ ...c, messages: c.messages.map(m => m.id === aiId ? { ...m, streaming: false } : m) })); setIsStreaming(false) },
      (err) => { setError(err); updateConvo(activeId, c => ({ ...c, messages: c.messages.filter(m => m.id !== aiId) })); setIsStreaming(false) },
      ctrl.signal
    )
  }

  const regenerate = async () => {
    if (isStreaming) return
    const lastUserIdx = [...messages].reverse().findIndex(m => m.role === 'user')
    if (lastUserIdx === -1) return
    const idx = messages.length - 1 - lastUserIdx
    const text = messages[idx].content
    updateConvo(activeId, c => ({ ...c, messages: c.messages.slice(0, idx + 1) }))
    await sendMessage(text)
  }

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input) }
  }

  const isEmpty = messages.length === 0
  const canSend = input.trim().length > 0 && !isStreaming

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg:     #07070d;
          --bg2:    #0d0d16;
          --bg3:    #12121e;
          --bg4:    #181828;
          --bgh:    #1e1e30;
          --b1:     rgba(255,255,255,0.03);
          --b2:     rgba(255,255,255,0.07);
          --b3:     rgba(255,255,255,0.12);
          --t1:     #eeeaff;
          --t2:     rgba(238,234,255,0.5);
          --t3:     rgba(238,234,255,0.22);
          --g1:     #5b6ef5;
          --g2:     #a78bfa;
          --g3:     #38bdf8;
          --g4:     #f472b6;
          --acc:    #a78bfa;
          --acc2:   #c4b5fd;
          --acd:    rgba(167,139,250,0.08);
          --green:  #4ade80;
          --red:    #f87171;
          --sw:     280px;
          --sw-col: 60px;
          font-family: 'Sora', sans-serif;
          -webkit-font-smoothing: antialiased;
        }

        html, body {
          width: 100%; height: 100%;
          margin: 0; padding: 0;
          overflow: hidden;
          background: var(--bg);
          color: var(--t1);
        }

        #root {
          width: 100vw;
          height: 100vh;
          display: flex;
        }

        button { cursor: pointer; border: none; background: none; color: inherit; font-family: inherit; }
        textarea { font-family: inherit; color: inherit; resize: none; border: none; background: none; outline: none; }

        /* ── Shell ── */
        .shell {
          display: flex;
          width: 100vw;
          height: 100vh;
          overflow: hidden;
          position: relative;
        }

        /* Ambient background glow */
        .shell::before {
          content: '';
          position: fixed;
          top: -30%;
          left: 10%;
          width: 60vw;
          height: 60vh;
          background: radial-gradient(ellipse, rgba(91,110,245,0.04) 0%, transparent 70%);
          pointer-events: none;
          z-index: 0;
        }
        .shell::after {
          content: '';
          position: fixed;
          bottom: -20%;
          right: 5%;
          width: 50vw;
          height: 50vh;
          background: radial-gradient(ellipse, rgba(167,139,250,0.05) 0%, transparent 70%);
          pointer-events: none;
          z-index: 0;
        }

        /* ── Sidebar ── */
        .sidebar {
          width: var(--sw);
          min-width: var(--sw);
          height: 100vh;
          background: var(--bg2);
          border-right: 1px solid var(--b1);
          display: flex;
          flex-direction: column;
          transition: width .25s cubic-bezier(.4,0,.2,1), min-width .25s cubic-bezier(.4,0,.2,1);
          overflow: hidden;
          position: relative;
          z-index: 10;
          flex-shrink: 0;
        }
        .sidebar.collapsed {
          width: var(--sw-col);
          min-width: var(--sw-col);
        }
        .sidebar.collapsed .hide-collapsed { display: none !important; }
        .sidebar.collapsed .sb-top { justify-content: center; padding: 16px 0; }
        .sidebar.collapsed .sb-footer { padding: 12px 0; justify-content: center; }

        .sb-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 20px 16px 16px;
          border-bottom: 1px solid var(--b1);
          flex-shrink: 0;
          gap: 8px;
        }

        .brand { display: flex; align-items: center; gap: 10px; overflow: hidden; }
        .brand-orb {
          width: 32px; height: 32px; min-width: 32px;
          border-radius: 10px;
          background: linear-gradient(135deg, var(--g1), var(--g2));
          display: flex; align-items: center; justify-content: center;
          color: white; flex-shrink: 0;
          box-shadow: 0 0 20px rgba(91,110,245,0.35);
        }
        .brand-name {
          font-size: 15px;
          font-weight: 700;
          letter-spacing: -.03em;
          white-space: nowrap;
          background: linear-gradient(90deg, var(--t1), var(--acc2));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .icon-btn {
          width: 32px; height: 32px; border-radius: 8px;
          display: flex; align-items: center; justify-content: center;
          color: var(--t3); transition: all .15s; flex-shrink: 0;
        }
        .icon-btn:hover { background: var(--bgh); color: var(--t1); }

        .new-btn {
          display: flex; align-items: center; gap: 8px;
          margin: 12px 12px 6px; padding: 9px 13px;
          border-radius: 10px;
          border: 1px solid var(--b2);
          font-size: 12.5px; font-weight: 500; color: var(--t2);
          transition: all .15s; white-space: nowrap;
          letter-spacing: -.01em;
        }
        .new-btn:hover {
          background: var(--acd);
          color: var(--acc2);
          border-color: rgba(167,139,250,0.3);
        }

        .conv-list {
          flex: 1; overflow-y: auto; padding: 8px 10px;
          scrollbar-width: thin;
          scrollbar-color: var(--b2) transparent;
        }
        .conv-list::-webkit-scrollbar { width: 2px; }
        .conv-list::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 2px; }

        .conv-label {
          font-size: 9px; font-weight: 600;
          letter-spacing: .12em; text-transform: uppercase;
          color: var(--t3); padding: 10px 6px 5px;
        }
        .conv-item {
          display: flex; flex-direction: column; gap: 2px;
          width: 100%; padding: 8px 10px; border-radius: 8px; text-align: left;
          border: 1px solid transparent; transition: all .12s; cursor: pointer;
        }
        .conv-item:hover { background: var(--bgh); }
        .conv-item.active {
          background: var(--acd);
          border-color: rgba(167,139,250,0.18);
        }
        .conv-title {
          font-size: 12px; font-weight: 500;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          color: var(--t1);
        }
        .conv-sub {
          font-size: 10px; color: var(--t3);
          font-family: 'Fira Code', monospace;
        }

        .sb-footer {
          border-top: 1px solid var(--b1);
          padding: 12px;
          flex-shrink: 0;
        }
        .user-row { display: flex; align-items: center; gap: 10px; }
        .av {
          width: 32px; height: 32px; min-width: 32px; border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 12px; font-weight: 700; flex-shrink: 0;
        }
        .av.user {
          background: linear-gradient(135deg, var(--bg4), var(--bgh));
          border: 1px solid var(--b3); color: var(--acc2);
        }
        .user-meta { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .user-name { font-size: 12.5px; font-weight: 600; letter-spacing: -.01em; }
        .user-plan { font-size: 10px; color: var(--t3); }

        /* ── Main ── */
        .main {
          flex: 1;
          display: flex;
          flex-direction: column;
          height: 100vh;
          overflow: hidden;
          background: var(--bg);
          position: relative;
          z-index: 1;
          min-width: 0;
        }

        /* ── Topbar ── */
        .topbar {
          height: 56px;
          display: flex; align-items: center; justify-content: space-between;
          padding: 0 24px;
          border-bottom: 1px solid var(--b1);
          flex-shrink: 0;
          background: rgba(7,7,13,0.8);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }
        .topbar-l, .topbar-r { display: flex; align-items: center; gap: 8px; }

        .model-pill {
          display: inline-flex; align-items: center; gap: 7px;
          font-size: 11px; font-weight: 500; color: var(--t2);
          background: var(--bg3); border: 1px solid var(--b2);
          padding: 5px 12px; border-radius: 999px;
          font-family: 'Fira Code', monospace;
          letter-spacing: -.01em;
        }
        .gem-dot {
          width: 7px; height: 7px; border-radius: 50%;
          background: conic-gradient(var(--g1), var(--g2), var(--g3), var(--g4), var(--g1));
          animation: spin 4s linear infinite; flex-shrink: 0;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .gen-pill {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 10.5px; font-weight: 500; color: var(--acc2);
          background: var(--acd); border: 1px solid rgba(167,139,250,0.2);
          padding: 4px 11px; border-radius: 999px;
          font-family: 'Fira Code', monospace;
        }
        .gpulse {
          width: 6px; height: 6px; border-radius: 50%; background: var(--acc2);
          animation: gp .9s ease-in-out infinite;
        }
        @keyframes gp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.2;transform:scale(.5)} }

        .stop-btn {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 11px; font-weight: 500; color: var(--red);
          padding: 5px 12px; border-radius: 999px;
          border: 1px solid rgba(248,113,113,0.2);
          background: rgba(248,113,113,0.05);
          transition: background .14s;
          font-family: 'Sora', sans-serif;
        }
        .stop-btn:hover { background: rgba(248,113,113,0.1); }

        /* ── Messages ── */
        .msgs-area {
          flex: 1;
          overflow-y: auto;
          padding: 0;
          scrollbar-width: thin;
          scrollbar-color: var(--b2) transparent;
        }
        .msgs-area::-webkit-scrollbar { width: 3px; }
        .msgs-area::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 3px; }

        .msgs-inner {
          width: 100%;
          max-width: 900px;
          margin: 0 auto;
          padding: 32px 32px 16px;
        }

        /* ── Welcome ── */
        .welcome {
          display: flex; flex-direction: column; align-items: center;
          gap: 16px;
          padding: 80px 32px 32px;
          text-align: center;
          animation: fadeUp .5s ease both;
          width: 100%;
          max-width: 600px;
          margin: 0 auto;
        }
        .welcome-orb {
          width: 84px; height: 84px; border-radius: 26px;
          background: linear-gradient(135deg, var(--g1), var(--g2), var(--g4));
          display: flex; align-items: center; justify-content: center;
          position: relative;
          box-shadow: 0 0 60px rgba(91,110,245,0.3), 0 0 120px rgba(167,139,250,0.15);
          color: white;
        }
        .welcome-orb svg { width: 34px; height: 34px; }
        .orb-ring {
          position: absolute; inset: -14px; border-radius: 34px;
          border: 1px solid rgba(167,139,250,0.15);
          animation: pulse 3.5s ease-in-out infinite;
        }
        .orb-ring.r2 {
          inset: -26px; border-radius: 42px;
          border-color: rgba(91,110,245,0.07);
          animation-delay: 1s;
        }
        .orb-ring.r3 {
          inset: -40px; border-radius: 54px;
          border-color: rgba(167,139,250,0.04);
          animation-delay: 2s;
        }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(1.03)} }

        .welcome h1 {
          font-size: clamp(2rem, 4vw, 3rem);
          font-weight: 300;
          letter-spacing: -.05em;
          line-height: 1.1;
          color: var(--t1);
          margin-top: 12px;
        }
        .welcome h1 em {
          font-style: normal;
          font-weight: 700;
          background: linear-gradient(135deg, var(--acc2), var(--g3));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .welcome p {
          font-size: 14px; color: var(--t3);
          max-width: 380px; line-height: 1.7;
          font-weight: 300;
        }
        .powered-badge {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 10.5px; color: var(--t3);
          background: var(--bg3); border: 1px solid var(--b2);
          padding: 4px 12px; border-radius: 999px;
          font-family: 'Fira Code', monospace;
          margin-top: 4px;
        }

        /* ── Msg rows ── */
        .msg-row {
          display: flex; gap: 14px;
          padding: 10px 0;
          animation: fadeUp .2s ease both;
          width: 100%;
        }
        .msg-row.user { flex-direction: row-reverse; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

        .msg-avatar {
          width: 34px; height: 34px; min-width: 34px; border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 12px; font-weight: 700; flex-shrink: 0;
          align-self: flex-start; margin-top: 2px;
        }
        .msg-avatar.ai {
          background: linear-gradient(135deg, var(--g1), var(--g2));
          color: #fff;
          box-shadow: 0 0 16px rgba(91,110,245,0.25);
        }
        .msg-avatar.user {
          background: var(--bg4); border: 1px solid var(--b3); color: var(--acc2);
        }

        .msg-body {
          display: flex; flex-direction: column; gap: 4px;
          max-width: 65%;
          min-width: 0;
        }
        .msg-row.assistant .msg-body { align-items: flex-start; margin-right: auto; }
        .msg-row.user .msg-body { align-items: flex-end; margin-left: auto; }

        .msg-bubble {
          padding: 13px 17px; border-radius: 16px;
          font-size: 14px; line-height: 1.8;
          border: 1px solid var(--b1);
          word-break: break-word;
        }
        .msg-bubble.assistant {
          background: var(--bg2);
          border-radius: 4px 16px 16px 16px;
          border-color: var(--b1);
        }
        .msg-bubble.user {
          background: linear-gradient(135deg, rgba(91,110,245,0.12), rgba(167,139,250,0.12));
          border-radius: 16px 4px 16px 16px;
          border-color: rgba(167,139,250,0.18);
        }

        .msg-footer { display: flex; align-items: center; gap: 6px; padding: 2px 2px 0; }
        .msg-time { font-size: 9.5px; color: var(--t3); font-family: 'Fira Code', monospace; }
        .msg-actions { display: flex; gap: 2px; opacity: 0; transition: opacity .14s; }
        .msg-row:hover .msg-actions { opacity: 1; }
        .action-btn {
          width: 24px; height: 24px; border-radius: 6px;
          display: flex; align-items: center; justify-content: center;
          color: var(--t3); transition: all .12s;
        }
        .action-btn:hover { background: var(--bgh); color: var(--t1); }

        .caret {
          display: inline-block; color: var(--acc2); font-size: 15px;
          animation: blink .75s step-start infinite; margin-left: 2px;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

        /* ── Markdown ── */
        .md { color: var(--t1); }
        .md p { margin: 0 0 10px; line-height: 1.8; }
        .md p:last-child { margin: 0; }
        .md strong { font-weight: 600; color: var(--acc2); }
        .md em { font-style: italic; color: var(--t2); }
        .md h1 { font-size: 1.15rem; font-weight: 700; margin: 16px 0 7px; letter-spacing: -.02em; }
        .md h2 { font-size: 1.05rem; font-weight: 600; margin: 14px 0 6px; }
        .md h3 { font-size: .95rem; font-weight: 600; margin: 12px 0 5px; color: var(--t2); }
        .md ul, .md ol { margin: 6px 0 10px 18px; display: flex; flex-direction: column; gap: 4px; }
        .md li { line-height: 1.7; }
        .md ul li { list-style: disc; }
        .md ol li { list-style: decimal; }
        .md code {
          font-family: 'Fira Code', monospace; font-size: 12.5px;
          background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.15);
          padding: 1px 6px; border-radius: 5px; color: var(--acc2);
        }
        .md pre {
          margin: 10px 0; border-radius: 12px;
          background: rgba(0,0,0,0.4); border: 1px solid var(--b2); overflow-x: auto;
        }
        .md pre code {
          display: block; padding: 14px 16px; background: none; border: none;
          color: #c4b5fd; font-size: 12.5px; line-height: 1.7; white-space: pre;
        }
        .md blockquote {
          margin: 10px 0; padding: 10px 16px;
          border-left: 2px solid var(--acc); background: var(--acd);
          border-radius: 0 10px 10px 0; color: var(--t2); font-style: italic;
        }
        .md hr { border: none; border-top: 1px solid var(--b2); margin: 14px 0; }
        .md a { color: var(--acc2); text-decoration: underline; }

        /* ── Error ── */
        .err-bar {
          display: flex; align-items: center; gap: 8px;
          padding: 10px 14px; border-radius: 10px;
          background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.18);
          font-size: 12.5px; color: var(--red);
          margin-bottom: 10px;
        }
        .err-bar button { margin-left: auto; color: var(--red); opacity: .5; }
        .err-bar button:hover { opacity: 1; }

        /* ── Suggestions ── */
        .suggestions-wrap {
          width: 100%;
          padding: 0 32px 24px;
        }
        .suggestions {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 10px;
          max-width: 900px;
          margin: 0 auto;
        }
        @media (max-width: 900px) {
          .suggestions { grid-template-columns: repeat(2, 1fr); }
        }

        .sugg-chip {
          display: flex; flex-direction: column; gap: 6px;
          padding: 14px 16px; border-radius: 12px;
          border: 1px solid var(--b2); background: var(--bg2);
          text-align: left; transition: all .15s; cursor: pointer;
          position: relative; overflow: hidden;
        }
        .sugg-chip::before {
          content: '';
          position: absolute; inset: 0;
          background: linear-gradient(135deg, rgba(91,110,245,0.04), rgba(167,139,250,0.04));
          opacity: 0; transition: opacity .15s;
        }
        .sugg-chip:hover::before { opacity: 1; }
        .sugg-chip:hover { border-color: rgba(167,139,250,0.25); transform: translateY(-2px); }
        .sugg-icon { font-size: 18px; line-height: 1; color: var(--acc); }
        .sugg-label {
          font-size: 9px; font-weight: 700;
          letter-spacing: .1em; text-transform: uppercase; color: var(--acc);
        }
        .sugg-text { font-size: 12px; color: var(--t2); line-height: 1.45; font-weight: 300; }

        /* ── Input ── */
        .input-area {
          padding: 12px 24px 20px;
          background: var(--bg);
          border-top: 1px solid var(--b1);
          flex-shrink: 0;
        }
        .input-inner {
          max-width: 900px;
          margin: 0 auto;
        }
        .ibox {
          display: flex; align-items: flex-end; gap: 8px;
          background: var(--bg2); border: 1px solid var(--b2);
          border-radius: 16px; padding: 10px 10px 10px 18px;
          transition: border-color .18s, box-shadow .18s;
        }
        .ibox:focus-within {
          border-color: rgba(167,139,250,0.35);
          box-shadow: 0 0 0 3px rgba(167,139,250,0.06), 0 4px 24px rgba(0,0,0,0.3);
        }
        .ibox textarea {
          flex: 1; font-size: 14px; line-height: 1.65;
          max-height: 180px; min-height: 22px;
          color: var(--t1);
        }
        .ibox textarea::placeholder { color: var(--t3); }
        .ibox textarea:disabled { opacity: .35; }

        .send-btn {
          width: 38px; height: 38px; border-radius: 11px;
          display: flex; align-items: center; justify-content: center;
          background: var(--bg4); color: var(--t3); border: 1px solid var(--b2);
          transition: all .14s; flex-shrink: 0;
        }
        .send-btn.ready {
          background: linear-gradient(135deg, var(--g1), var(--g2));
          color: #fff; border-color: transparent;
          box-shadow: 0 4px 20px rgba(91,110,245,0.35);
        }
        .send-btn.ready:hover { filter: brightness(1.1); transform: scale(1.05); }
        .send-btn:disabled { opacity: .2; cursor: default; }

        .input-hint {
          font-size: 10px; color: var(--t3); text-align: center; margin-top: 8px;
          font-family: 'Fira Code', monospace; letter-spacing: .03em;
        }

        /* ── Responsive ── */
        @media (max-width: 700px) {
          .sidebar { position: fixed; left: 0; top: 0; bottom: 0; z-index: 100; }
          .sidebar.collapsed { width: 0; min-width: 0; }
          .msgs-inner { padding: 20px 16px 12px; }
          .suggestions-wrap { padding: 0 16px 16px; }
          .input-area { padding: 10px 16px 16px; }
          .msg-body { max-width: 85%; }
        }
      `}</style>

      <div className="shell">
        {/* ── Sidebar ── */}
        <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sb-top">
            <div className="brand hide-collapsed">
              <div className="brand-orb"><StarMark /></div>
              <span className="brand-name">Princess AI</span>
            </div>
            <div className="brand-orb" style={{ display: sidebarOpen ? 'none' : 'flex' }}><StarMark /></div>
            <button className="icon-btn hide-collapsed" onClick={() => setSidebarOpen(o => !o)} aria-label="Toggle sidebar">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18"/>
              </svg>
            </button>
          </div>

          <button className="new-btn hide-collapsed" onClick={newConversation}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            New conversation
          </button>

          <nav className="conv-list hide-collapsed" aria-label="Conversations">
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

          <div className="sb-footer hide-collapsed">
            <div className="user-row">
              <div className="av user">Y</div>
              <div className="user-meta">
                <span className="user-name">You</span>
                <span className="user-plan">Princess Free</span>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="main">
          {/* Topbar */}
          <header className="topbar">
            <div className="topbar-l">
              {!sidebarOpen && (
                <button className="icon-btn" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 12h18M3 6h18M3 18h18"/>
                  </svg>
                </button>
              )}
              <div className="model-pill">
                <div className="gem-dot"/>
                gemini-2.5-flash
              </div>
              {isStreaming && (
                <div className="gen-pill">
                  <div className="gpulse"/>
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
              <button className="icon-btn" onClick={newConversation} aria-label="New conversation">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 5v14M5 12h14"/>
                </svg>
              </button>
            </div>
          </header>

          {/* Messages */}
          <div className="msgs-area">
            {isEmpty ? (
              <div className="welcome">
                <div className="welcome-orb">
                  <div className="orb-ring"/>
                  <div className="orb-ring r2"/>
                  <div className="orb-ring r3"/>
                  <StarMark />
                </div>
                <h1>Hello, I'm <em>Princess</em></h1>
                <p>Ask me anything — I'll wander around the answer and find it eventually.</p>
                <div className="powered-badge">
                  <div className="gem-dot"/>
                  Powered by Gemini 2.5 Flash
                </div>
              </div>
            ) : (
              <div className="msgs-inner">
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
                <div ref={bottomRef} style={{ height: 8 }}/>
              </div>
            )}
            {isEmpty && error && (
              <div style={{ padding: '0 32px' }}>
                <div className="err-bar" style={{ maxWidth: 900, margin: '0 auto' }}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
                  </svg>
                  {error}
                  <button onClick={() => setError(null)}>✕</button>
                </div>
              </div>
            )}
          </div>

          {/* Suggestions */}
          {isEmpty && (
            <div className="suggestions-wrap">
              <div className="suggestions">
                {SUGGESTIONS.map(s => (
                  <button key={s.text} className="sugg-chip" onClick={() => sendMessage(s.text)}>
                    <span className="sugg-icon">{s.icon}</span>
                    <span className="sugg-label">{s.label}</span>
                    <span className="sugg-text">{s.text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="input-area">
            <div className="input-inner">
              <div className="ibox">
                <textarea
                  ref={taRef}
                  value={input}
                  rows={1}
                  placeholder="Message Princess…"
                  aria-label="Message input"
                  onChange={e => { setInput(e.target.value); autoResize() }}
                  onKeyDown={handleKey}
                  disabled={isStreaming}
                />
                <button
                  className={`send-btn ${canSend ? 'ready' : ''}`}
                  onClick={() => sendMessage(input)}
                  disabled={!canSend}
                  title="Send"
                  aria-label="Send message"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z"/>
                  </svg>
                </button>
              </div>
              <p className="input-hint">Enter to send · Shift+Enter for new line</p>
            </div>
          </div>
        </main>
      </div>
    </>
  )
}