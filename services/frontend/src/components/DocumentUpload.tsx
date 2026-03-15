import { useRef, useState, useCallback, useEffect } from 'react';
import { api } from '../api/client';
import type { UploadResult } from '../types';
import {
  Upload,
  X,
  FileText,
  CheckCircle,
  AlertCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
  RefreshCw,
} from 'lucide-react';
import clsx from 'clsx';

interface Props {
  conversationId: string;
}

type FileStatus = 'pending' | 'uploading' | 'done' | 'error';

interface FileEntry {
  id: string;
  filename: string;
  /** Defined only for locally-added (not yet uploaded) files */
  file?: File;
  status: FileStatus;
  chunks?: number;
  error?: string;
}

function makeId() {
  return Math.random().toString(36).slice(2);
}

export function DocumentUpload({ conversationId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loadingExisting, setLoadingExisting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const ACCEPTED = ['.pdf', '.docx', '.doc'];

  // ── Load existing files from MinIO whenever the conversation changes ────────
  useEffect(() => {
    let cancelled = false;
    setEntries([]);
    setLoadingExisting(true);

    api
      .listDocuments(conversationId)
      .then(({ files }) => {
        if (cancelled) return;
        setEntries(
          files.map((filename) => ({
            id: makeId(),
            filename,
            status: 'done',
          })),
        );
        if (files.length > 0) setExpanded(true);
      })
      .catch(() => {
        // Non-fatal — just show empty list
      })
      .finally(() => {
        if (!cancelled) setLoadingExisting(false);
      });

    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  // ── Add local files (drag-drop or browse) ──────────────────────────────────
  function addFiles(newFiles: File[]) {
    const valid = newFiles.filter((f) =>
      ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    // Skip files that are already listed (by name)
    const existing = new Set(entries.map((e) => e.filename));
    const fresh = valid.filter((f) => !existing.has(f.name));
    if (!fresh.length) return;
    setEntries((prev) => [
      ...prev,
      ...fresh.map((f) => ({
        id: makeId(),
        filename: f.name,
        file: f,
        status: 'pending' as FileStatus,
      })),
    ]);
    setExpanded(true);
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  }, [entries]);

  // ── Upload pending files ────────────────────────────────────────────────────
  async function handleUpload() {
    const pending = entries.filter((e) => e.status === 'pending' && e.file);
    if (!pending.length || uploading) return;
    setUploading(true);

    setEntries((prev) =>
      prev.map((e) =>
        e.status === 'pending' ? { ...e, status: 'uploading' } : e,
      ),
    );

    try {
      const res = await api.uploadDocuments(
        conversationId,
        pending.map((e) => e.file!),
      );
      setEntries((prev) =>
        prev.map((entry) => {
          if (entry.status !== 'uploading') return entry;
          const result: UploadResult | undefined = res.uploaded.find(
            (r) => r.filename === entry.filename,
          );
          if (!result) return entry;
          const ok = result.status === 'indexed' || result.status === 'empty_document';
          return {
            ...entry,
            status: ok ? ('done' as FileStatus) : ('error' as FileStatus),
            chunks: result.chunks,
            error: ok ? undefined : result.status,
          };
        }),
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      setEntries((prev) =>
        prev.map((e) =>
          e.status === 'uploading' ? { ...e, status: 'error', error: msg } : e,
        ),
      );
    } finally {
      setUploading(false);
    }
  }

  function removeEntry(id: string) {
    setEntries((prev) => prev.filter((e) => e.id !== id));
  }

  function formatSize(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  const pendingCount = entries.filter((e) => e.status === 'pending').length;
  const doneCount = entries.filter((e) => e.status === 'done').length;

  return (
    <div className="border-t border-slate-700/50 bg-slate-800/30">
      {/* Toggle bar */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-700/30 transition-colors text-sm"
      >
        <div className="flex items-center gap-2 text-slate-400">
          {loadingExisting ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <Upload size={14} />
          )}
          <span className="font-medium">
            Documents
            {doneCount > 0 && (
              <span className="ml-2 text-xs text-emerald-400 font-normal">
                {doneCount} indexed
              </span>
            )}
            {pendingCount > 0 && (
              <span className="ml-2 text-xs text-amber-400 font-normal">
                {pendingCount} pending
              </span>
            )}
          </span>
        </div>
        {expanded ? (
          <ChevronDown size={14} className="text-slate-500" />
        ) : (
          <ChevronUp size={14} className="text-slate-500" />
        )}
      </button>

      {/* Expanded panel */}
      {expanded && (
        <div className="px-4 pb-4 space-y-3 animate-fade-in">
          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={clsx(
              'border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all',
              dragging
                ? 'border-indigo-500 bg-indigo-500/10'
                : 'border-slate-600 hover:border-slate-500 hover:bg-slate-700/30',
            )}
          >
            <Upload
              size={20}
              className={clsx(
                'mx-auto mb-2',
                dragging ? 'text-indigo-400' : 'text-slate-500',
              )}
            />
            <p className="text-sm text-slate-400">
              Drop files here or{' '}
              <span className="text-indigo-400 font-medium">browse</span>
            </p>
            <p className="text-xs text-slate-600 mt-1">PDF, DOCX, DOC</p>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.doc"
              className="hidden"
              onChange={(e) => addFiles(Array.from(e.target.files ?? []))}
            />
          </div>

          {/* File list */}
          {loadingExisting ? (
            <div className="flex items-center justify-center gap-2 py-3 text-slate-500 text-xs">
              <Loader2 size={12} className="animate-spin" />
              Loading uploaded files…
            </div>
          ) : entries.length > 0 ? (
            <div className="space-y-1.5 max-h-44 overflow-y-auto">
              {entries.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center gap-2.5 p-2 rounded-lg bg-slate-700/40 text-sm"
                >
                  <FileText size={14} className="text-slate-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-slate-200 truncate text-xs font-medium">
                      {entry.filename}
                    </p>
                    <p className="text-slate-500 text-xs">
                      {entry.file ? formatSize(entry.file.size) : 'stored'}
                      {entry.chunks != null && entry.status === 'done' && (
                        <span className="ml-2 text-emerald-400">
                          {entry.chunks} chunks
                        </span>
                      )}
                      {entry.error && (
                        <span className="ml-2 text-rose-400">{entry.error}</span>
                      )}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {entry.status === 'uploading' && (
                      <Loader2 size={14} className="text-indigo-400 animate-spin" />
                    )}
                    {entry.status === 'done' && (
                      <CheckCircle size={14} className="text-emerald-400" />
                    )}
                    {entry.status === 'error' && (
                      <AlertCircle size={14} className="text-rose-400" />
                    )}
                    {/* Only allow removing locally-added (not yet uploaded) entries */}
                    {(entry.status === 'pending' || entry.status === 'error') && (
                      <button
                        onClick={() => removeEntry(entry.id)}
                        className="text-slate-500 hover:text-rose-400 transition-colors"
                        title="Remove"
                      >
                        <X size={13} />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-xs text-slate-600 py-2">
              No documents uploaded yet.
            </p>
          )}

          {/* Action buttons row */}
          <div className="flex gap-2">
            {pendingCount > 0 && (
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="flex-1 flex items-center justify-center gap-2 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium rounded-xl transition-colors"
              >
                {uploading ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Uploading…
                  </>
                ) : (
                  <>
                    <Upload size={14} />
                    Upload {pendingCount} file{pendingCount > 1 ? 's' : ''}
                  </>
                )}
              </button>
            )}
            {/* Refresh button — re-fetches MinIO listing */}
            {!uploading && (
              <button
                onClick={() => {
                  setLoadingExisting(true);
                  api
                    .listDocuments(conversationId)
                    .then(({ files }) => {
                      setEntries((prev) => {
                        const pending = prev.filter(
                          (e) => e.status === 'pending' || e.status === 'uploading',
                        );
                        const persisted = files.map((filename) => ({
                          id: makeId(),
                          filename,
                          status: 'done' as FileStatus,
                        }));
                        // Merge: keep pending local files, replace done entries
                        const pendingNames = new Set(pending.map((e) => e.filename));
                        return [
                          ...persisted.filter((e) => !pendingNames.has(e.filename)),
                          ...pending,
                        ];
                      });
                    })
                    .catch(() => {})
                    .finally(() => setLoadingExisting(false));
                }}
                className={clsx(
                  'p-2 rounded-xl text-slate-500 hover:text-slate-300 hover:bg-slate-700/50 transition-colors',
                  pendingCount === 0 ? 'flex-1 flex items-center justify-center gap-1.5 text-xs' : '',
                )}
                title="Refresh file list"
              >
                <RefreshCw size={14} className={loadingExisting ? 'animate-spin' : ''} />
                {pendingCount === 0 && <span>Refresh</span>}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
