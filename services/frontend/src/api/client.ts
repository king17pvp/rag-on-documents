import type {
  Conversation,
  ConversationHistory,
  HealthStatus,
  UploadResult,
} from '../types';

const BASE = '/api/v1';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health(): Promise<HealthStatus> {
    return request(`${BASE}/health`);
  },

  createConversation(userId: string, title?: string): Promise<Conversation> {
    return request(`${BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, title: title || undefined }),
    });
  },

  getHistory(conversationId: string): Promise<ConversationHistory> {
    return request(`${BASE}/conversations/${conversationId}/history`);
  },

  chat(
    conversationId: string,
    userId: string,
    question: string,
  ): Promise<{ conversation_id: string; response: string; response_type: string }> {
    return request(`${BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conversationId,
        user_id: userId,
        question,
      }),
    });
  },

  uploadDocuments(
    conversationId: string,
    files: File[],
  ): Promise<{ conversation_id: string; uploaded: UploadResult[] }> {
    const form = new FormData();
    form.append('conversation_id', conversationId);
    files.forEach((f) => form.append('files', f));
    return request(`${BASE}/documents/upload`, { method: 'POST', body: form });
  },

  listDocuments(
    conversationId: string,
  ): Promise<{ conversation_id: string; files: string[] }> {
    return request(`${BASE}/conversations/${conversationId}/documents`);
  },
};
