import { useState } from "react";
import { API_BASE_URL } from "../App";
import Button from "../components/Button";
import ErrorDisplay from "../components/ErrorDisplay";
import JsonDisplay from "../components/JsonDisplay";
import Input from "../components/Input";

const Verification = () => {
  const [url, setUrl] = useState("");
  const [keepAudio, setKeepAudio] = useState(false);
  const [isLoadingVideo, setIsLoadingVideo] = useState(false);
  const [videoResult, setVideoResult] = useState(null);
  const [videoError, setVideoError] = useState(null);

  const [content, setContent] = useState("");
  const [isLoadingContent, setIsLoadingContent] = useState(false);
  const [contentResult, setContentResult] = useState(null);
  const [contentError, setContentError] = useState(null);

  const handleVideoSubmit = async (endpoint) => {
    setIsLoadingVideo(true);
    setVideoError(null);
    setVideoResult(null);

    try {
      const body =
        endpoint === "full-pipeline"
          ? JSON.stringify({ url, keep_audio: keepAudio })
          : JSON.stringify({ url });

      const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      const data = await response.json();
      if (!response.ok) throw data;

      setVideoResult(data);
    } catch (err) {
      setVideoError(err);
    } finally {
      setIsLoadingVideo(false);
    }
  };

  const handleContentSubmit = async () => {
    setIsLoadingContent(true);
    setContentError(null);
    setContentResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });

      const data = await response.json();
      if (!response.ok) throw data;

      setContentResult(data);
    } catch (err) {
      setContentError(err);
    } finally {
      setIsLoadingContent(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* --- Video Verification --- */}
      <div className="card bg-base-100 shadow-md border border-base-200">
        <div className="card-body space-y-4">
          <h2 className="card-title text-primary">🎥 Video Verification</h2>

          <div>
            <label htmlFor="video-url" className="label">
              <span className="label-text font-medium">
                Video URL (YouTube, etc.)
              </span>
            </label>
            <Input
              id="video-url"
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <Button
              onClick={() => handleVideoSubmit("verify-video")}
              isLoading={isLoadingVideo}
              disabled={!url}
            >
              Simple Verify
            </Button>
            <Button
              onClick={() => handleVideoSubmit("full-pipeline")}
              isLoading={isLoadingVideo}
              disabled={!url}
              variant="secondary"
            >
              Full Pipeline
            </Button>
          </div>

          <div className="form-control">
            <label className="cursor-pointer label justify-start gap-3">
              <input
                type="checkbox"
                checked={keepAudio}
                onChange={(e) => setKeepAudio(e.target.checked)}
                className="checkbox checkbox-primary"
              />
              <span className="label-text">
                Keep audio file (Full Pipeline only)
              </span>
            </label>
          </div>

          <ErrorDisplay error={videoError} />
          {videoResult && <JsonDisplay data={videoResult} />}
        </div>
      </div>

      {/* --- Content Verification --- */}
      <div className="card bg-base-100 shadow-md border border-base-200">
        <div className="card-body space-y-4">
          <h2 className="card-title text-primary">🧾 Content Verification</h2>

          <div>
            <label htmlFor="text-content" className="label">
              <span className="label-text font-medium">Text / Fact to Verify</span>
            </label>
            <Input
              id="text-content"
              type="text"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter the fact or text to verify..."
            />
          </div>

          <Button
            onClick={handleContentSubmit}
            isLoading={isLoadingContent}
            disabled={!content}
          >
            Verify Content
          </Button>

          <ErrorDisplay error={contentError} />
          {contentResult && <JsonDisplay data={contentResult} />}
        </div>
      </div>
    </div>
  );
};

export default Verification;
