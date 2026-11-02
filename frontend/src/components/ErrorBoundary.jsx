import React from 'react'

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          minHeight: '100vh', background: '#0d1117', padding: '2rem',
        }}>
          <div style={{
            background: '#161b22', border: '1px solid #f8514930',
            borderRadius: 16, padding: '2rem 2.5rem', maxWidth: 540, textAlign: 'center',
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '0.6rem' }}>⚠️</div>
            <h2 style={{ color: '#f85149', fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.6rem' }}>
              Something went wrong
            </h2>
            <p style={{ color: '#8b949e', fontSize: '0.85rem', lineHeight: 1.6, marginBottom: '1rem' }}>
              {this.state.error?.message || 'An unexpected error occurred.'}
            </p>
            <button
              onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload() }}
              style={{
                background: '#58a6ff', color: '#000', border: 'none', borderRadius: 8,
                padding: '8px 20px', fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer',
              }}
            >
              Reload Dashboard
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
