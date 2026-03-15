import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Conversation, ChatMessage } from '../types';

// Fixed user seeded in the database via devops/init.sql
const DEFAULT_USER_ID = '00000000-0000-0000-0000-000000000001';

interface AppState {
  userId: string;
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: Record<string, ChatMessage[]>;

  setCurrentConversation: (id: string | null) => void;
  addConversation: (conv: Conversation) => void;
  removeConversation: (id: string) => void;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  updateMessage: (
    conversationId: string,
    messageId: string,
    update: Partial<ChatMessage>,
  ) => void;
  setMessages: (conversationId: string, messages: ChatMessage[]) => void;
  clearMessages: (conversationId: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      userId: DEFAULT_USER_ID,
      conversations: [],
      currentConversationId: null,
      messages: {},

      setCurrentConversation: (id) => set({ currentConversationId: id }),

      addConversation: (conv) =>
        set((state) => ({
          conversations: [conv, ...state.conversations],
        })),

      removeConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.filter(
            (c) => c.conversation_id !== id,
          ),
          currentConversationId:
            state.currentConversationId === id
              ? (state.conversations.find((c) => c.conversation_id !== id)
                  ?.conversation_id ?? null)
              : state.currentConversationId,
          messages: Object.fromEntries(
            Object.entries(state.messages).filter(([k]) => k !== id),
          ),
        })),

      addMessage: (conversationId, message) =>
        set((state) => ({
          messages: {
            ...state.messages,
            [conversationId]: [
              ...(state.messages[conversationId] ?? []),
              message,
            ],
          },
        })),

      updateMessage: (conversationId, messageId, update) =>
        set((state) => ({
          messages: {
            ...state.messages,
            [conversationId]: (state.messages[conversationId] ?? []).map((m) =>
              m.id === messageId ? { ...m, ...update } : m,
            ),
          },
        })),

      setMessages: (conversationId, messages) =>
        set((state) => ({
          messages: { ...state.messages, [conversationId]: messages },
        })),

      clearMessages: (conversationId) =>
        set((state) => ({
          messages: { ...state.messages, [conversationId]: [] },
        })),
    }),
    {
      name: 'chatwithdoc-storage',
      partialize: (state) => ({
        conversations: state.conversations,
        currentConversationId: state.currentConversationId,
      }),
    },
  ),
);
