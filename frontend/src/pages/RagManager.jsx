import { useCallback, useEffect, useState } from "react";
import { API_BASE_URL } from "../App";
import Button from "../components/Button";
import ErrorDisplay from "../components/ErrorDisplay";
import FileDropzone from "../components/FileDropzone";
import Input from "../components/Input";
import { Protect } from '@clerk/clerk-react'
import JsonDisplay from "../components/JsonDisplay";
import {
  Trash2,
  RefreshCw,
  Bot,
  Send,
} from "lucide-react";

const RagManager = () => {
  const [pdfFiles, setPdfFiles] = useState([]);
  const [imageFiles, setImageFiles] = useState([]);
  const [audioFile, setAudioFile] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState("");
  const [question, setQuestion] = useState("");
  const [isLoadingCreate, setIsLoadingCreate] = useState(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [createResult, setCreateResult] = useState(null);
  const [chatResult, setChatResult] = useState(null);
  const [error, setError] = useState(null);
  const [chatError, setChatError] = useState(null);

  // Fetch sessions
  const fetchSessions = useCallback(async () => {
    setIsLoadingSessions(true);
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`);
      const data = await response.json();
      if (!response.ok) throw data;
      setSessions(data.sessions || []);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Create session
  const handleCreateSession = async () => {
    setIsLoadingCreate(true);
    setError(null);
    setCreateResult(null);

    const formData = new FormData();
    pdfFiles.forEach((file) => formData.append("pdf_files", file));
    imageFiles.forEach((file) => formData.append("image_files", file));
    if (audioFile) formData.append("audio_file", audioFile);

    try {
      const response = await fetch(`${API_BASE_URL}/create-rag-session`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw data;
      setCreateResult(data);
      setActiveSession(data.session_id);
      fetchSessions();
      setPdfFiles([]);
      setImageFiles([]);
      setAudioFile(null);
    } catch (err) {
      setError(err);
    } finally {
      setIsLoadingCreate(false);
    }
  };

  // Delete session
  const handleDeleteSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/delete-session`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      const data = await response.json();
      if (!response.ok) throw data;
      fetchSessions();
      if (activeSession === sessionId) setActiveSession("");
    } catch (err) {
      setError(err);
    }
  };

  // Chat
  const handleChat = async () => {
    if (!activeSession || !question) return;

    setIsLoadingChat(true);
    setChatError(null);
    setChatResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: activeSession, question }),
      });
      const data = await response.json();
      if (!response.ok) throw data;
      setChatResult(data);
      setQuestion("");
    } catch (err) {
      setChatError(err);
    } finally {
      setIsLoadingChat(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* --- Left Column: Create & Manage Sessions --- */}
      <div className="space-y-6">
        {/* Create Session */}
        <div className="card bg-base-100 shadow-xl p-6 rounded-xl space-y-4">
          <h2 className="text-xl font-bold text-primary">
            1️⃣ Create RAG Session
          </h2>
          <div className="space-y-3">
          {/* <Protect plan="premium"> */}
            <label className="font-medium">PDFs</label>
            <FileDropzone
              onFilesChange={(f) => setPdfFiles(f)}
              multiple
              accept=".pdf"
            />
            <label className="font-medium">Images</label>
            <FileDropzone
              onFilesChange={(f) => setImageFiles(f)}
              multiple
              accept="image/*"
            />

            <label className="font-medium">Audio</label>
            <FileDropzone
              onFilesChange={(f) => setAudioFile(f[0])}
              accept="audio/*"
            />
          </div>

          <Button
            onClick={handleCreateSession}
            isLoading={isLoadingCreate}
            disabled={!pdfFiles.length && !imageFiles.length && !audioFile}
          >
            Create Session
          </Button>

          <ErrorDisplay error={error} />
          {createResult && <JsonDisplay data={createResult} />}
        </div>

        {/* Manage Sessions */}
        <div className="card bg-base-100 shadow-xl p-6 rounded-xl space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold text-primary">
              2️⃣ Active Sessions
            </h2>
            <Button
              onClick={fetchSessions}
              isLoading={isLoadingSessions}
              variant="secondary"
              size="sm"
            >
              <RefreshCw size={16} />
            </Button>
          </div>

          <div className="max-h-64 overflow-y-auto space-y-2 pr-1">
            {sessions.length > 0 ? (
              sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`flex justify-between items-center p-3 rounded-lg border transition-all ${
                    activeSession === session.session_id
                      ? "bg-primary/10 border-primary text-primary"
                      : "hover:bg-base-200 border-base-300"
                  }`}
                >
                  <div className="text-sm overflow-hidden">
                    <span
                      className="font-mono cursor-pointer hover:underline"
                      onClick={() => setActiveSession(session.session_id)}
                    >
                      {session.session_id.substring(0, 13)}...
                    </span>
                    <span className="block text-xs opacity-70">
                      {new Date(session.created_at).toLocaleString()}
                    </span>
                  </div>
                  <Button
                    onClick={() => handleDeleteSession(session.session_id)}
                    variant="danger"
                  >
                    <Trash2 size={16} />
                  </Button>
                </div>
              ))
            ) : (
              <p className="text-sm opacity-70">No active sessions.</p>
            )}
          </div>
        </div>
      </div>

      {/* --- Right Column: Chat --- */}
      <div className="lg:col-span-2 card bg-base-100 shadow-xl p-6 rounded-xl space-y-4">
        <h2 className="text-xl font-bold text-primary">3️⃣ Chat with Session</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Active Session ID
            </label>
            <Input
              type="text"
              placeholder="Select or create a session"
              value={activeSession}
              onChange={(e) => setActiveSession(e.target.value)}
            />
          </div>

          {chatResult && (
            <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
              <div className="flex gap-3">
                <Bot className="w-6 h-6 text-primary shrink-0" />
                <div>
                  <h3 className="font-semibold text-primary">Answer</h3>
                  <p className="text-base-content whitespace-pre-wrap">
                    {chatResult.answer}
                  </p>
                  <span className="text-xs opacity-70">
                    (Sources: {chatResult.sources_count} | Model:{" "}
                    {chatResult.model})
                  </span>
                </div>
              </div>
            </div>
          )}

          <ErrorDisplay error={chatError} />

          {/* Chat Input */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              type="text"
              placeholder="Ask a question about your documents..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleChat()}
              disabled={!activeSession}
            />
            <Button
              onClick={handleChat}
              isLoading={isLoadingChat}
              disabled={!activeSession || !question}
            >
              <Send size={18} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RagManager;
