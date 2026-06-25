"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ViewId =
  | "home"
  | "new"
  | "project"
  | "score"
  | "mixer"
  | "form"
  | "validation"
  | "datasets"
  | "export";

type FileRecord = {
  kind: string;
  path: string;
  status?: string;
  bytes?: number;
  reason?: string;
  track_id?: string;
};

type TrackSummary = {
  id: string;
  instrument: string;
  role: string;
  bars: number;
  metadata: Record<string, unknown>;
};

type ProjectSummary = {
  project_id: string;
  metadata: Record<string, unknown>;
  bar_count: number;
  tracks: TrackSummary[];
};

type ValidationIssue = {
  severity: "error" | "warning";
  validator: string;
  code: string;
  message: string;
  track_id?: string | null;
  bar_number?: number | null;
  beat?: number | null;
};

type ValidationReport = {
  status?: string;
  errors?: ValidationIssue[];
  warnings?: ValidationIssue[];
  metrics?: Record<string, unknown>;
};

type GenerationSpec = {
  style?: string;
  key?: string;
  tempo?: number;
  form?: string;
  ensemble?: string;
  seed?: number;
  density?: string;
};

type ProjectResponse = {
  project_id: string;
  status: string;
  project: ProjectSummary;
  generation_spec?: GenerationSpec | null;
  export_manifest?: {
    status?: string;
    files?: FileRecord[];
    pdf_status?: string;
  };
  validation?: ValidationReport;
};

type GenerateResponse = {
  project_id: string;
  status: string;
  project: ProjectSummary;
  files: FileRecord[];
  validation: ValidationReport;
};

type DatasetSummary = {
  imported_files?: number;
  duplicate_files?: number;
  extracted_patterns?: number;
  pattern_counts?: Record<string, number>;
};

type DatasetRecord = {
  dataset_id: string;
  summary: DatasetSummary;
  pattern_index_path: string;
};

type PatternRecord = {
  id: string;
  category: string;
  role: string;
  style: string;
  quality: number;
  source_file_id: string;
  usable_for_training: boolean;
  usable_for_pattern_extraction: boolean;
  payload: Record<string, unknown>;
};

type ChordEntry = {
  symbol: string;
  bar?: number;
  beat?: number;
};

type SectionEntry = {
  name: string;
  start_bar: number;
  end_bar: number;
  label?: string | null;
};

type ProjectJson = {
  chord_grid?: ChordEntry[];
  form?: SectionEntry[];
};

type MixerState = Record<string, { mute: boolean; solo: boolean; volume: number }>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_PROMPT =
  "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria";
const QUICK_PRESETS = [
  {
    label: "Hard bop sextet",
    seed: 1234,
    prompt:
      "hard bop nocturno en Do menor, 132 bpm, blues menor, sexteto con saxo alto, trompeta, trombon, piano, contrabajo y bateria, shout chorus final",
  },
  {
    label: "Bebop blues",
    seed: 2401,
    prompt:
      "bebop blues en Fa mayor, 204 bpm, quinteto con saxo alto, trompeta, piano, contrabajo y bateria, denso y energico",
  },
  {
    label: "Swing AABA",
    seed: 2402,
    prompt:
      "swing medio en Si bemol mayor, 144 bpm, AABA de 32 compases, cuarteto con saxo tenor, piano, contrabajo y bateria",
  },
  {
    label: "Ballad quartet",
    seed: 2403,
    prompt:
      "jazz ballad lirica en Fa mayor, 72 bpm, AABA, cuarteto con saxo tenor, piano, contrabajo y bateria, poco denso",
  },
  {
    label: "Modal quintet",
    seed: 2404,
    prompt:
      "modal jazz en Re menor, 124 bpm, vamp modal de 16 compases, quinteto con saxo alto, trompeta, piano, contrabajo y bateria",
  },
  {
    label: "Bossa quartet",
    seed: 2405,
    prompt:
      "bossa nova en Sol mayor, 138 bpm, forma latin 32, cuarteto con saxo alto, piano, contrabajo y bateria, relajado",
  },
  {
    label: "Jazz waltz",
    seed: 2406,
    prompt: "jazz waltz en La menor, 150 bpm, 3/4, trio con piano, contrabajo y bateria, lirico",
  },
  {
    label: "Funk jazz",
    seed: 2407,
    prompt:
      "funk jazz straight-eighth en Mi menor, 108 bpm, vamp de 16 compases, quinteto con saxo alto, trompeta, piano, contrabajo y bateria",
  },
];

const views: { id: ViewId; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "new", label: "New project" },
  { id: "project", label: "Project detail" },
  { id: "score", label: "Score viewer" },
  { id: "mixer", label: "Mixer" },
  { id: "form", label: "Chord/form" },
  { id: "validation", label: "Validation" },
  { id: "datasets", label: "Datasets" },
  { id: "export", label: "Export" },
];

export default function Home() {
  const [activeView, setActiveView] = useState<ViewId>("home");
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [seed, setSeed] = useState(1234);
  const [includePdf, setIncludePdf] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [files, setFiles] = useState<FileRecord[]>([]);
  const [validation, setValidation] = useState<ValidationReport>({});
  const [compiledSpec, setCompiledSpec] = useState<GenerationSpec | null>(null);
  const [projectJson, setProjectJson] = useState<ProjectJson>({});
  const [musicXml, setMusicXml] = useState("");
  const [scoreStatus, setScoreStatus] = useState("");
  const [mixer, setMixer] = useState<MixerState>({});
  const [regenerateInstruction, setRegenerateInstruction] = useState("menos movimiento");
  const [datasetId, setDatasetId] = useState("local-library");
  const [datasetSource, setDatasetSource] = useState("");
  const [datasetStyle, setDatasetStyle] = useState("hard_bop");
  const [datasetLicense, setDatasetLicense] = useState("CC0-1.0");
  const [datasetQuality, setDatasetQuality] = useState(4);
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [patternCategory, setPatternCategory] = useState("walking_bass_cells");
  const [patternRole, setPatternRole] = useState("walking_bass");
  const [patterns, setPatterns] = useState<PatternRecord[]>([]);
  const [localChords, setLocalChords] = useState<ChordEntry[]>([]);

  const selectedProjectId = projectId || project?.project_id || "";
  const scoreRef = useRef<HTMLDivElement | null>(null);
  const validationIssues = useMemo(
    () => [...(validation.errors ?? []), ...(validation.warnings ?? [])],
    [validation],
  );
  const exportedFiles = files.length ? files : project?.export_manifest?.files ?? [];
  const statusText =
    apiStatus === "online" ? "API online" : apiStatus === "offline" ? "API offline" : "Checking";

  useEffect(() => {
    void checkHealth();
    void loadDatasets();
  }, []);

  useEffect(() => {
    const tracks = project?.project.tracks ?? [];
    setMixer((current) => {
      const next: MixerState = {};
      for (const track of tracks) {
        next[track.id] = current[track.id] ?? { mute: false, solo: false, volume: 82 };
      }
      return next;
    });
  }, [project]);

  useEffect(() => {
    setLocalChords(projectJson.chord_grid ?? []);
  }, [projectJson]);

  useEffect(() => {
    if (!musicXml || activeView !== "score" || scoreRef.current === null) {
      return;
    }
    let cancelled = false;
    const element = scoreRef.current;
    element.innerHTML = "";
    setScoreStatus("Rendering");
    import("opensheetmusicdisplay")
      .then(async ({ OpenSheetMusicDisplay }) => {
        const osmd = new OpenSheetMusicDisplay(element, {
          autoResize: true,
          backend: "svg",
          drawTitle: true,
        });
        await osmd.load(musicXml);
        if (!cancelled) {
          await osmd.render();
          setScoreStatus("Ready");
        }
      })
      .catch((error: unknown) => {
        setScoreStatus(error instanceof Error ? error.message : "Score render failed");
      });
    return () => {
      cancelled = true;
    };
  }, [activeView, musicXml]);

  async function checkHealth() {
    try {
      await requestJson<{ status: string }>("/health");
      setApiStatus("online");
    } catch {
      setApiStatus("offline");
    }
  }

  async function compilePrompt() {
    setBusy(true);
    setMessage("Compiling prompt");
    try {
      const spec = await requestJson<GenerationSpec>("/v1/prompts/compile", {
        method: "POST",
        body: JSON.stringify({ prompt, seed }),
      });
      setCompiledSpec(spec);
      setMessage("Prompt compiled");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function generateProject() {
    setBusy(true);
    setMessage("Generating project");
    try {
      const generated = await requestJson<GenerateResponse>("/v1/projects/generate", {
        method: "POST",
        body: JSON.stringify({
          prompt,
          seed,
          options: { export: true, validate: true, include_pdf: includePdf },
        }),
      });
      setProjectId(generated.project_id);
      setFiles(generated.files);
      setValidation(generated.validation);
      await loadProject(generated.project_id);
      await loadProjectArtifacts(generated.project_id);
      setActiveView("project");
      setMessage("Project generated");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function loadProject(id = selectedProjectId) {
    if (!id) {
      return;
    }
    const loaded = await requestJson<ProjectResponse>(`/v1/projects/${encodeURIComponent(id)}`);
    setProject(loaded);
    setProjectId(loaded.project_id);
    setValidation(loaded.validation ?? {});
    setFiles(loaded.export_manifest?.files ?? []);
  }

  async function loadProjectArtifacts(id = selectedProjectId) {
    if (!id) {
      return;
    }
    try {
      const projectText = await requestText(
        `/v1/projects/${encodeURIComponent(id)}/file?kind=project`,
      );
      setProjectJson(JSON.parse(projectText) as ProjectJson);
    } catch {
      setProjectJson({});
    }
    try {
      const xml = await requestText(`/v1/projects/${encodeURIComponent(id)}/file?kind=musicxml`);
      setMusicXml(xml);
    } catch {
      setMusicXml("");
      setScoreStatus("No exported score");
    }
  }

  async function exportProject() {
    if (!selectedProjectId) {
      return;
    }
    setBusy(true);
    setMessage("Exporting project");
    try {
      const exported = await requestJson<{
        status: string;
        files: FileRecord[];
        validation: ValidationReport;
      }>(`/v1/projects/${encodeURIComponent(selectedProjectId)}/export`, {
        method: "POST",
        body: JSON.stringify({ include_pdf: includePdf }),
      });
      setFiles(exported.files);
      setValidation(exported.validation);
      await loadProject(selectedProjectId);
      await loadProjectArtifacts(selectedProjectId);
      setActiveView("export");
      setMessage("Export ready");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function refreshValidation() {
    if (!selectedProjectId) {
      return;
    }
    setBusy(true);
    setMessage("Validating project");
    try {
      const report = await requestJson<ValidationReport>(
        `/v1/projects/${encodeURIComponent(selectedProjectId)}/validation`,
      );
      setValidation(report);
      setActiveView("validation");
      setMessage("Validation refreshed");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function regenerateProject() {
    if (!selectedProjectId) {
      return;
    }
    setBusy(true);
    setMessage("Regenerating project");
    try {
      const regenerated = await requestJson<GenerateResponse>(
        `/v1/projects/${encodeURIComponent(selectedProjectId)}/regenerate`,
        {
          method: "POST",
          body: JSON.stringify({
            target: { track: "piano", bars: [9, 10, 11, 12] },
            instruction: regenerateInstruction,
            seed: seed + 1,
            options: { validate: true, export: true, include_pdf: includePdf },
          }),
        },
      );
      setFiles(regenerated.files);
      setValidation(regenerated.validation);
      await loadProject(selectedProjectId);
      await loadProjectArtifacts(selectedProjectId);
      setMessage("Project regenerated");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function loadDatasets() {
    try {
      const payload = await requestJson<{ datasets: DatasetRecord[] }>("/v1/datasets");
      setDatasets(payload.datasets);
    } catch {
      setDatasets([]);
    }
  }

  async function importDataset() {
    setBusy(true);
    setMessage("Importing dataset");
    try {
      await requestJson("/v1/datasets/import", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: datasetId,
          source_dir: datasetSource,
          default_metadata: {
            source: "web",
            license: datasetLicense,
            copyright_notes: "Imported from studio UI",
            usable_for_training: true,
            usable_for_pattern_extraction: true,
            style: datasetStyle,
            quality: datasetQuality,
            tags: ["drums", "walking_bass", "piano", "melody", "horn_response"],
          },
        }),
      });
      await loadDatasets();
      setMessage("Dataset imported");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  async function searchPatterns() {
    setBusy(true);
    setMessage("Searching patterns");
    try {
      const params = new URLSearchParams({
        category: patternCategory,
        role: patternRole,
        style: datasetStyle,
        usable_for_pattern_extraction: "true",
      });
      if (datasetId) {
        params.set("dataset_id", datasetId);
      }
      const payload = await requestJson<{ patterns: PatternRecord[] }>(
        `/v1/patterns/search?${params.toString()}`,
      );
      setPatterns(payload.patterns);
      setMessage("Patterns loaded");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  function playGuidePreview() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || localChords.length === 0) {
      return;
    }
    const context = new AudioContextClass();
    const now = context.currentTime + 0.05;
    const chords = localChords.slice(0, 12);
    chords.forEach((chord, index) => {
      const frequency = chordFrequency(chord.symbol);
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "triangle";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, now + index * 0.32);
      gain.gain.exponentialRampToValueAtTime(0.08, now + index * 0.32 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.32 + 0.26);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(now + index * 0.32);
      oscillator.stop(now + index * 0.32 + 0.28);
    });
  }

  function updateMixer(trackId: string, update: Partial<MixerState[string]>) {
    setMixer((current) => ({
      ...current,
      [trackId]: { ...(current[trackId] ?? { mute: false, solo: false, volume: 82 }), ...update },
    }));
  }

  function updateChord(index: number, symbol: string) {
    setLocalChords((current) =>
      current.map((chord, chordIndex) =>
        chordIndex === index ? { ...chord, symbol } : chord,
      ),
    );
  }

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">AI Arranger Studio</p>
          <h1>Symbolic studio</h1>
        </div>
        <div className="topbar-actions">
          <span className={`status-strip ${apiStatus}`}>{statusText}</span>
          <input
            aria-label="Project id"
            className="project-id-input"
            onChange={(event) => setProjectId(event.target.value)}
            placeholder="project id"
            value={projectId}
          />
          <button className="secondary-button" onClick={() => void loadProject()} type="button">
            Load
          </button>
        </div>
      </header>

      <nav className="view-tabs" aria-label="Studio sections">
        {views.map((view) => (
          <button
            className={activeView === view.id ? "tab-button active" : "tab-button"}
            key={view.id}
            onClick={() => setActiveView(view.id)}
            type="button"
          >
            {view.label}
          </button>
        ))}
      </nav>

      <section className="status-row" aria-live="polite">
        <span>{message || "Ready"}</span>
        {busy ? <span className="busy-pill">Working</span> : null}
      </section>

      {activeView === "home" ? (
        <HomeView
          busy={busy}
          files={exportedFiles}
          generateProject={generateProject}
          project={project}
          setActiveView={setActiveView}
          validation={validation}
        />
      ) : null}

      {activeView === "new" ? (
        <NewProjectView
          busy={busy}
          compilePrompt={compilePrompt}
          compiledSpec={compiledSpec}
          generateProject={generateProject}
          includePdf={includePdf}
          prompt={prompt}
          seed={seed}
          setIncludePdf={setIncludePdf}
          setPrompt={setPrompt}
          setSeed={setSeed}
        />
      ) : null}

      {activeView === "project" ? (
        <ProjectDetailView
          project={project}
          regenerateInstruction={regenerateInstruction}
          regenerateProject={regenerateProject}
          setActiveView={setActiveView}
          setRegenerateInstruction={setRegenerateInstruction}
        />
      ) : null}

      {activeView === "score" ? (
        <ScoreView
          exportProject={exportProject}
          musicXml={musicXml}
          playGuidePreview={playGuidePreview}
          scoreRef={scoreRef}
          scoreStatus={scoreStatus}
          selectedProjectId={selectedProjectId}
        />
      ) : null}

      {activeView === "mixer" ? (
        <MixerView mixer={mixer} project={project} updateMixer={updateMixer} />
      ) : null}

      {activeView === "form" ? (
        <FormEditorView
          chords={localChords}
          form={projectJson.form ?? []}
          regenerateProject={regenerateProject}
          updateChord={updateChord}
        />
      ) : null}

      {activeView === "validation" ? (
        <ValidationView
          issues={validationIssues}
          refreshValidation={refreshValidation}
          validation={validation}
        />
      ) : null}

      {activeView === "datasets" ? (
        <DatasetsView
          datasetId={datasetId}
          datasetLicense={datasetLicense}
          datasetQuality={datasetQuality}
          datasetSource={datasetSource}
          datasetStyle={datasetStyle}
          datasets={datasets}
          importDataset={importDataset}
          patternCategory={patternCategory}
          patternRole={patternRole}
          patterns={patterns}
          searchPatterns={searchPatterns}
          setDatasetId={setDatasetId}
          setDatasetLicense={setDatasetLicense}
          setDatasetQuality={setDatasetQuality}
          setDatasetSource={setDatasetSource}
          setDatasetStyle={setDatasetStyle}
          setPatternCategory={setPatternCategory}
          setPatternRole={setPatternRole}
        />
      ) : null}

      {activeView === "export" ? (
        <ExportView
          exportProject={exportProject}
          files={exportedFiles}
          includePdf={includePdf}
          selectedProjectId={selectedProjectId}
          setIncludePdf={setIncludePdf}
        />
      ) : null}
    </main>
  );
}

function HomeView({
  busy,
  files,
  generateProject,
  project,
  setActiveView,
  validation,
}: {
  busy: boolean;
  files: FileRecord[];
  generateProject: () => Promise<void>;
  project: ProjectResponse | null;
  setActiveView: (view: ViewId) => void;
  validation: ValidationReport;
}) {
  return (
    <section className="dashboard-grid">
      <div className="command-panel">
        <p className="eyebrow">Home</p>
        <h2>{project ? project.project.project_id : "No active project"}</h2>
        <div className="metric-grid">
          <Metric label="Bars" value={project?.project.bar_count ?? "-"} />
          <Metric label="Tracks" value={project?.project.tracks.length ?? "-"} />
          <Metric label="Validation" value={validation.status ?? "-"} />
          <Metric label="Files" value={files.length || "-"} />
        </div>
        <div className="button-row">
          <button disabled={busy} onClick={() => void generateProject()} type="button">
            Generate
          </button>
          <button className="secondary-button" onClick={() => setActiveView("score")} type="button">
            Score
          </button>
          <button className="secondary-button" onClick={() => setActiveView("export")} type="button">
            Export
          </button>
        </div>
      </div>
      <TrackTable tracks={project?.project.tracks ?? []} />
    </section>
  );
}

function NewProjectView({
  busy,
  compilePrompt,
  compiledSpec,
  generateProject,
  includePdf,
  prompt,
  seed,
  setIncludePdf,
  setPrompt,
  setSeed,
}: {
  busy: boolean;
  compilePrompt: () => Promise<void>;
  compiledSpec: GenerationSpec | null;
  generateProject: () => Promise<void>;
  includePdf: boolean;
  prompt: string;
  seed: number;
  setIncludePdf: (value: boolean) => void;
  setPrompt: (value: string) => void;
  setSeed: (value: number) => void;
}) {
  return (
    <section className="workspace-grid">
      <div className="command-panel">
        <p className="eyebrow">New project</p>
        <h2>Prompt</h2>
        <div className="preset-strip" aria-label="Generation presets">
          {QUICK_PRESETS.map((preset) => (
            <button
              className="preset-button"
              key={preset.label}
              onClick={() => {
                setPrompt(preset.prompt);
                setSeed(preset.seed);
              }}
              type="button"
            >
              {preset.label}
            </button>
          ))}
        </div>
        <textarea
          aria-label="Arrangement prompt"
          onChange={(event) => setPrompt(event.target.value)}
          spellCheck={false}
          value={prompt}
        />
        <div className="inline-controls">
          <label>
            <span>Seed</span>
            <input
              min={0}
              onChange={(event) => setSeed(Number(event.target.value || 0))}
              type="number"
              value={seed}
            />
          </label>
          <label className="toggle-row">
            <input
              checked={includePdf}
              onChange={(event) => setIncludePdf(event.target.checked)}
              type="checkbox"
            />
            <span>PDF</span>
          </label>
        </div>
        <div className="button-row">
          <button className="secondary-button" disabled={busy} onClick={compilePrompt} type="button">
            Compile
          </button>
          <button disabled={busy} onClick={generateProject} type="button">
            Generate
          </button>
        </div>
      </div>
      <div className="data-panel">
        <p className="eyebrow">GenerationSpec</p>
        <SpecGrid spec={compiledSpec} />
      </div>
    </section>
  );
}

function ProjectDetailView({
  project,
  regenerateInstruction,
  regenerateProject,
  setActiveView,
  setRegenerateInstruction,
}: {
  project: ProjectResponse | null;
  regenerateInstruction: string;
  regenerateProject: () => Promise<void>;
  setActiveView: (view: ViewId) => void;
  setRegenerateInstruction: (value: string) => void;
}) {
  return (
    <section className="workspace-grid">
      <div className="data-panel wide-panel">
        <p className="eyebrow">Project detail</p>
        <h2>{project?.project.project_id ?? "No project loaded"}</h2>
        <SpecGrid spec={project?.generation_spec ?? null} />
        <TrackTable tracks={project?.project.tracks ?? []} />
      </div>
      <div className="command-panel">
        <p className="eyebrow">Regenerate</p>
        <h2>Targeted pass</h2>
        <textarea
          aria-label="Regeneration instruction"
          className="compact-textarea"
          onChange={(event) => setRegenerateInstruction(event.target.value)}
          value={regenerateInstruction}
        />
        <div className="button-row">
          <button onClick={regenerateProject} type="button">
            Regenerate
          </button>
          <button className="secondary-button" onClick={() => setActiveView("validation")} type="button">
            Validation
          </button>
        </div>
      </div>
    </section>
  );
}

function ScoreView({
  exportProject,
  musicXml,
  playGuidePreview,
  scoreRef,
  scoreStatus,
  selectedProjectId,
}: {
  exportProject: () => Promise<void>;
  musicXml: string;
  playGuidePreview: () => void;
  scoreRef: React.RefObject<HTMLDivElement | null>;
  scoreStatus: string;
  selectedProjectId: string;
}) {
  return (
    <section className="score-layout">
      <div className="score-toolbar">
        <p className="eyebrow">Score viewer</p>
        <div className="button-row">
          <button className="secondary-button" onClick={exportProject} type="button">
            Export
          </button>
          <button className="secondary-button" onClick={playGuidePreview} type="button">
            Play preview
          </button>
          {selectedProjectId ? (
            <a
              className="link-button"
              href={fileUrl(selectedProjectId, "midi")}
              rel="noreferrer"
              target="_blank"
            >
              Open MIDI
            </a>
          ) : null}
        </div>
      </div>
      <div className="score-surface" ref={scoreRef}>
        {musicXml ? null : <span>{scoreStatus || "No score"}</span>}
      </div>
      <p className="panel-note">{scoreStatus || "Ready"}</p>
    </section>
  );
}

function MixerView({
  mixer,
  project,
  updateMixer,
}: {
  mixer: MixerState;
  project: ProjectResponse | null;
  updateMixer: (trackId: string, update: Partial<MixerState[string]>) => void;
}) {
  return (
    <section className="data-panel">
      <p className="eyebrow">Track mixer</p>
      <h2>{project?.project.tracks.length ?? 0} tracks</h2>
      <div className="mixer-grid">
        {(project?.project.tracks ?? []).map((track) => {
          const state = mixer[track.id] ?? { mute: false, solo: false, volume: 82 };
          return (
            <div className="mixer-row" key={track.id}>
              <div>
                <strong>{track.id}</strong>
                <span>{track.role}</span>
              </div>
              <label className="toggle-row">
                <input
                  checked={state.mute}
                  onChange={(event) => updateMixer(track.id, { mute: event.target.checked })}
                  type="checkbox"
                />
                <span>Mute</span>
              </label>
              <label className="toggle-row">
                <input
                  checked={state.solo}
                  onChange={(event) => updateMixer(track.id, { solo: event.target.checked })}
                  type="checkbox"
                />
                <span>Solo</span>
              </label>
              <label className="volume-control">
                <span>{state.volume}</span>
                <input
                  max={100}
                  min={0}
                  onChange={(event) => updateMixer(track.id, { volume: Number(event.target.value) })}
                  type="range"
                  value={state.volume}
                />
              </label>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function FormEditorView({
  chords,
  form,
  regenerateProject,
  updateChord,
}: {
  chords: ChordEntry[];
  form: SectionEntry[];
  regenerateProject: () => Promise<void>;
  updateChord: (index: number, symbol: string) => void;
}) {
  return (
    <section className="workspace-grid">
      <div className="data-panel wide-panel">
        <p className="eyebrow">Chord/form editor</p>
        <h2>{chords.length} chord cells</h2>
        <div className="chord-grid">
          {chords.slice(0, 64).map((chord, index) => (
            <label className="chord-cell" key={`${chord.bar}-${chord.beat}-${index}`}>
              <span>
                {chord.bar ?? "-"}:{chord.beat ?? 1}
              </span>
              <input
                aria-label={`Chord ${index + 1}`}
                onChange={(event) => updateChord(index, event.target.value)}
                value={chord.symbol}
              />
            </label>
          ))}
        </div>
      </div>
      <div className="data-panel">
        <p className="eyebrow">Form</p>
        <div className="section-list">
          {form.map((section) => (
            <div className="section-row" key={`${section.name}-${section.start_bar}`}>
              <strong>{section.label || section.name}</strong>
              <span>
                {section.start_bar}-{section.end_bar}
              </span>
            </div>
          ))}
        </div>
        <button onClick={regenerateProject} type="button">
          Regenerate
        </button>
      </div>
    </section>
  );
}

function ValidationView({
  issues,
  refreshValidation,
  validation,
}: {
  issues: ValidationIssue[];
  refreshValidation: () => Promise<void>;
  validation: ValidationReport;
}) {
  return (
    <section className="data-panel">
      <div className="panel-heading-row">
        <div>
          <p className="eyebrow">Validation report</p>
          <h2>{validation.status ?? "No report"}</h2>
        </div>
        <button className="secondary-button" onClick={refreshValidation} type="button">
          Refresh
        </button>
      </div>
      <div className="issue-table">
        <div className="table-row header-row">
          <span>Severity</span>
          <span>Validator</span>
          <span>Track</span>
          <span>Bar</span>
          <span>Message</span>
        </div>
        {issues.length ? (
          issues.map((issue, index) => (
            <div className="table-row" key={`${issue.code}-${index}`}>
              <span className={issue.severity}>{issue.severity}</span>
              <span>{issue.validator}</span>
              <span>{issue.track_id ?? "-"}</span>
              <span>{issue.bar_number ?? "-"}</span>
              <span>{issue.message}</span>
            </div>
          ))
        ) : (
          <div className="empty-state">No issues</div>
        )}
      </div>
    </section>
  );
}

function DatasetsView({
  datasetId,
  datasetLicense,
  datasetQuality,
  datasetSource,
  datasetStyle,
  datasets,
  importDataset,
  patternCategory,
  patternRole,
  patterns,
  searchPatterns,
  setDatasetId,
  setDatasetLicense,
  setDatasetQuality,
  setDatasetSource,
  setDatasetStyle,
  setPatternCategory,
  setPatternRole,
}: {
  datasetId: string;
  datasetLicense: string;
  datasetQuality: number;
  datasetSource: string;
  datasetStyle: string;
  datasets: DatasetRecord[];
  importDataset: () => Promise<void>;
  patternCategory: string;
  patternRole: string;
  patterns: PatternRecord[];
  searchPatterns: () => Promise<void>;
  setDatasetId: (value: string) => void;
  setDatasetLicense: (value: string) => void;
  setDatasetQuality: (value: number) => void;
  setDatasetSource: (value: string) => void;
  setDatasetStyle: (value: string) => void;
  setPatternCategory: (value: string) => void;
  setPatternRole: (value: string) => void;
}) {
  return (
    <section className="workspace-grid">
      <div className="command-panel">
        <p className="eyebrow">Dataset library</p>
        <h2>Import</h2>
        <div className="form-stack">
          <label>
            <span>Dataset id</span>
            <input onChange={(event) => setDatasetId(event.target.value)} value={datasetId} />
          </label>
          <label>
            <span>Source folder</span>
            <input onChange={(event) => setDatasetSource(event.target.value)} value={datasetSource} />
          </label>
          <label>
            <span>Style</span>
            <input onChange={(event) => setDatasetStyle(event.target.value)} value={datasetStyle} />
          </label>
          <label>
            <span>License</span>
            <input
              onChange={(event) => setDatasetLicense(event.target.value)}
              value={datasetLicense}
            />
          </label>
          <label>
            <span>Quality</span>
            <input
              max={5}
              min={1}
              onChange={(event) => setDatasetQuality(Number(event.target.value))}
              type="number"
              value={datasetQuality}
            />
          </label>
        </div>
        <button onClick={importDataset} type="button">
          Import
        </button>
      </div>
      <div className="data-panel">
        <p className="eyebrow">Datasets</p>
        <div className="dataset-list">
          {datasets.map((dataset) => (
            <button
              className="dataset-row"
              key={dataset.dataset_id}
              onClick={() => setDatasetId(dataset.dataset_id)}
              type="button"
            >
              <strong>{dataset.dataset_id}</strong>
              <span>{dataset.summary.extracted_patterns ?? 0} patterns</span>
            </button>
          ))}
        </div>
        <div className="pattern-search">
          <input
            aria-label="Pattern category"
            onChange={(event) => setPatternCategory(event.target.value)}
            value={patternCategory}
          />
          <input
            aria-label="Pattern role"
            onChange={(event) => setPatternRole(event.target.value)}
            value={patternRole}
          />
          <button className="secondary-button" onClick={searchPatterns} type="button">
            Search
          </button>
        </div>
        <div className="pattern-list">
          {patterns.map((pattern) => (
            <div className="pattern-row" key={pattern.id}>
              <strong>{pattern.id}</strong>
              <span>{pattern.category}</span>
              <span>{pattern.role}</span>
              <span>Q{pattern.quality}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ExportView({
  exportProject,
  files,
  includePdf,
  selectedProjectId,
  setIncludePdf,
}: {
  exportProject: () => Promise<void>;
  files: FileRecord[];
  includePdf: boolean;
  selectedProjectId: string;
  setIncludePdf: (value: boolean) => void;
}) {
  return (
    <section className="data-panel">
      <div className="panel-heading-row">
        <div>
          <p className="eyebrow">Export panel</p>
          <h2>{files.length} files</h2>
        </div>
        <div className="button-row">
          <label className="toggle-row">
            <input
              checked={includePdf}
              onChange={(event) => setIncludePdf(event.target.checked)}
              type="checkbox"
            />
            <span>PDF</span>
          </label>
          <button onClick={exportProject} type="button">
            Export
          </button>
          {selectedProjectId ? (
            <a className="link-button" href={zipUrl(selectedProjectId)}>
              Download ZIP
            </a>
          ) : null}
        </div>
      </div>
      <div className="file-table">
        <div className="table-row header-row">
          <span>Kind</span>
          <span>Track</span>
          <span>Status</span>
          <span>Bytes</span>
          <span>Open</span>
        </div>
        {files.map((file) => (
          <div className="table-row" key={`${file.kind}-${file.track_id ?? file.path}`}>
            <span>{file.kind}</span>
            <span>{file.track_id ?? "-"}</span>
            <span>{file.status ?? "created"}</span>
            <span>{file.bytes ?? "-"}</span>
            <span>
              {selectedProjectId && file.status !== "skipped" ? (
                <a
                  href={fileUrl(selectedProjectId, file.kind, file.track_id)}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open
                </a>
              ) : (
                "-"
              )}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SpecGrid({ spec }: { spec: GenerationSpec | null }) {
  const items = [
    ["Style", spec?.style ?? "-"],
    ["Key", spec?.key ?? "-"],
    ["Tempo", spec?.tempo ?? "-"],
    ["Form", spec?.form ?? "-"],
    ["Ensemble", spec?.ensemble ?? "-"],
    ["Density", spec?.density ?? "-"],
    ["Seed", spec?.seed ?? "-"],
  ];
  return (
    <div className="spec-grid">
      {items.map(([label, value]) => (
        <Metric key={label} label={String(label)} value={String(value)} />
      ))}
    </div>
  );
}

function TrackTable({ tracks }: { tracks: TrackSummary[] }) {
  return (
    <div className="data-panel">
      <p className="eyebrow">Tracks</p>
      <div className="track-table">
        <div className="table-row header-row">
          <span>Track</span>
          <span>Instrument</span>
          <span>Role</span>
          <span>Bars</span>
        </div>
        {tracks.map((track) => (
          <div className="table-row" key={track.id}>
            <span>{track.id}</span>
            <span>{track.instrument}</span>
            <span>{track.role}</span>
            <span>{track.bars}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

async function requestText(path: string): Promise<string> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.text();
}

function fileUrl(projectId: string, kind: string, trackId?: string): string {
  const params = new URLSearchParams({ kind });
  if (trackId) {
    params.set("track_id", trackId);
  }
  return `${API_BASE}/v1/projects/${encodeURIComponent(projectId)}/file?${params.toString()}`;
}

function zipUrl(projectId: string): string {
  return `${API_BASE}/v1/projects/${encodeURIComponent(projectId)}/zip`;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed";
}

function chordFrequency(symbol: string): number {
  const match = symbol.match(/^[A-G](?:#|b)?/);
  const root = match?.[0] ?? "C";
  const midiByRoot: Record<string, number> = {
    C: 48,
    "C#": 49,
    Db: 49,
    D: 50,
    "D#": 51,
    Eb: 51,
    E: 52,
    F: 53,
    "F#": 54,
    Gb: 54,
    G: 55,
    "G#": 56,
    Ab: 56,
    A: 57,
    "A#": 58,
    Bb: 58,
    B: 59,
  };
  const midi = midiByRoot[root] ?? 48;
  return 440 * 2 ** ((midi - 69) / 12);
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}
