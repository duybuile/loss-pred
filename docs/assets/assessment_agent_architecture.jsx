const palette = {
  ink: "#162033",
  muted: "#5C677D",
  line: "#BBC5D6",
  panel: "#F7F4EC",
  panelAlt: "#EAF3FF",
  panelWarm: "#FFF0D8",
  panelAlert: "#FCE5E1",
  accent: "#1C5D99",
  accentSoft: "#D8EAFB",
  success: "#2D6A4F",
  caution: "#C17C0E",
  danger: "#B54434",
  white: "#FFFFFF",
};

const typography = {
  display: "Avenir Next, Futura, Trebuchet MS, sans-serif",
  body: "IBM Plex Sans, Segoe UI, sans-serif",
  mono: "IBM Plex Mono, Consolas, monospace",
};

const lanes = [
  {
    title: "Input Record",
    subtitle: "Incoming risk submission",
    x: 72,
    y: 180,
    width: 210,
    height: 156,
    fill: palette.panel,
    bullets: [
      "risk_type, territory, industry",
      "limit, premium, prior_claims",
      "years_trading, broker",
    ],
  },
  {
    title: "Predict Loss",
    subtitle: "Primary evidence tool",
    x: 320,
    y: 132,
    width: 250,
    height: 252,
    fill: palette.panelAlt,
    bullets: [
      "probability_of_loss",
      "top_features",
      "input_quality_flags",
      "model_warnings",
    ],
  },
  {
    title: "Hybrid Controller",
    subtitle: "Deterministic policy engine",
    x: 620,
    y: 92,
    width: 360,
    height: 332,
    fill: palette.white,
    bullets: [
      "Compute confidence from |pl - 0.5|",
      "Downgrade for missing or invalid inputs",
      "Trigger retrieval only for low confidence",
      "Escalate if evidence conflicts",
    ],
  },
  {
    title: "Fallback Retrieval",
    subtitle: "Historical analogs on demand",
    x: 1038,
    y: 132,
    width: 248,
    height: 252,
    fill: palette.panelWarm,
    bullets: [
      "Run once for weak model evidence",
      "Use small n_results",
      "Summarize support or tension",
      "Never search for certainty",
    ],
  },
  {
    title: "Reviewer Response",
    subtitle: "Actionable assessment output",
    x: 1328,
    y: 180,
    width: 214,
    height: 156,
    fill: palette.panel,
    bullets: [
      "recommendation",
      "risk_assessment",
      "confidence_level",
      "review_guidance",
    ],
  },
];

const policyCards = [
  {
    title: "Confidence Policy",
    x: 640,
    y: 470,
    width: 322,
    height: 176,
    fill: palette.accentSoft,
    lines: [
      "High: margin >= 0.30",
      "Medium: 0.15 to 0.29",
      "Low: margin < 0.15",
      "Missing fields or warnings force low",
    ],
  },
  {
    title: "Guardrails",
    x: 1000,
    y: 470,
    width: 360,
    height: 176,
    fill: "#FDEEE5",
    lines: [
      "Bound retrieval to one round",
      "No repeated tool loops",
      "Escalate when model and retrieval disagree",
      "Use LLM only for coherent summaries",
    ],
  },
];

function Panel({ title, subtitle, x, y, width, height, fill, bullets }) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx="24"
        fill={fill}
        stroke={palette.line}
        strokeWidth="2"
      />
      <text
        x={x + 20}
        y={y + 34}
        fill={palette.ink}
        fontFamily={typography.display}
        fontSize="24"
        fontWeight="700"
      >
        {title}
      </text>
      <text
        x={x + 20}
        y={y + 60}
        fill={palette.muted}
        fontFamily={typography.body}
        fontSize="15"
        fontWeight="500"
      >
        {subtitle}
      </text>
      {bullets.map((bullet, index) => (
        <g key={`${title}-${bullet}`}>
          <circle
            cx={x + 24}
            cy={y + 92 + index * 26}
            r="4"
            fill={palette.accent}
          />
          <text
            x={x + 38}
            y={y + 97 + index * 26}
            fill={palette.ink}
            fontFamily={typography.body}
            fontSize="15"
          >
            {bullet}
          </text>
        </g>
      ))}
    </g>
  );
}

function PolicyCard({ title, x, y, width, height, fill, lines }) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx="22"
        fill={fill}
        stroke={palette.line}
        strokeWidth="2"
      />
      <text
        x={x + 20}
        y={y + 34}
        fill={palette.ink}
        fontFamily={typography.display}
        fontSize="22"
        fontWeight="700"
      >
        {title}
      </text>
      {lines.map((line, index) => (
        <text
          key={`${title}-${line}`}
          x={x + 20}
          y={y + 72 + index * 25}
          fill={palette.ink}
          fontFamily={typography.body}
          fontSize="15"
        >
          {line}
        </text>
      ))}
    </g>
  );
}

function Arrow({ path, color = palette.accent, width = 4, dashed = false }) {
  return (
    <path
      d={path}
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeDasharray={dashed ? "10 10" : undefined}
      markerEnd="url(#arrowhead)"
    />
  );
}

export default function AssessmentAgentArchitectureDiagram() {
  return (
    <svg
      viewBox="0 0 1600 900"
      role="img"
      aria-labelledby="assessment-agent-title assessment-agent-desc"
      xmlns="http://www.w3.org/2000/svg"
    >
      <title id="assessment-agent-title">Assessment Agent Architecture</title>
      <desc id="assessment-agent-desc">
        Diagram of a hybrid controller that scores risk with a model first, falls
        back to retrieval for low-confidence cases, and produces a reviewer-facing
        response with deterministic escalation rules.
      </desc>

      <defs>
        <linearGradient id="canvasGlow" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FFF9F0" />
          <stop offset="55%" stopColor="#F4F8FE" />
          <stop offset="100%" stopColor="#EEF6F1" />
        </linearGradient>
        <marker
          id="arrowhead"
          markerWidth="12"
          markerHeight="12"
          refX="8"
          refY="6"
          orient="auto"
        >
          <path d="M0,0 L12,6 L0,12 z" fill={palette.accent} />
        </marker>
        <marker
          id="alertArrow"
          markerWidth="12"
          markerHeight="12"
          refX="8"
          refY="6"
          orient="auto"
        >
          <path d="M0,0 L12,6 L0,12 z" fill={palette.danger} />
        </marker>
      </defs>

      <rect width="1600" height="900" fill="url(#canvasGlow)" />
      <circle cx="130" cy="96" r="64" fill="#FFE4C4" opacity="0.55" />
      <circle cx="1490" cy="130" r="84" fill="#D8EAFB" opacity="0.9" />
      <circle cx="1460" cy="810" r="110" fill="#E6F4EA" opacity="0.9" />

      <text
        x="72"
        y="82"
        fill={palette.ink}
        fontFamily={typography.display}
        fontSize="42"
        fontWeight="800"
      >
        Assessment Agent Architecture
      </text>
      <text
        x="72"
        y="118"
        fill={palette.muted}
        fontFamily={typography.body}
        fontSize="19"
      >
        Model-first reviewer support with fallback-only retrieval and deterministic escalation.
      </text>

      <rect
        x="72"
        y="714"
        width="1450"
        height="118"
        rx="28"
        fill={palette.white}
        stroke={palette.line}
        strokeWidth="2"
      />
      <text
        x="98"
        y="754"
        fill={palette.ink}
        fontFamily={typography.display}
        fontSize="24"
        fontWeight="700"
      >
        Output Contract
      </text>
      <text
        x="98"
        y="790"
        fill={palette.ink}
        fontFamily={typography.body}
        fontSize="16"
      >
        recommendation, risk_assessment, confidence_level, key_factors, similar_records_summary,
      </text>
      <text
        x="98"
        y="815"
        fill={palette.ink}
        fontFamily={typography.body}
        fontSize="16"
      >
        second_opinion_recommended, review_guidance, tools_used, warnings
      </text>

      {lanes.map((lane) => (
        <Panel key={lane.title} {...lane} />
      ))}

      {policyCards.map((card) => (
        <PolicyCard key={card.title} {...card} />
      ))}

      <Arrow path="M282 258 C304 258, 304 258, 320 258" />
      <Arrow path="M570 258 C594 258, 600 258, 620 258" />
      <Arrow path="M980 258 C1002 258, 1016 258, 1038 258" />
      <Arrow path="M1286 258 C1302 258, 1312 258, 1328 258" />

      <path
        d="M810 424 C810 452, 810 452, 810 470"
        fill="none"
        stroke={palette.success}
        strokeWidth="4"
        strokeLinecap="round"
        markerEnd="url(#arrowhead)"
      />
      <text
        x="836"
        y="452"
        fill={palette.success}
        fontFamily={typography.body}
        fontSize="14"
        fontWeight="700"
      >
        confidence policy
      </text>

      <path
        d="M980 258 C1050 258, 1070 390, 1108 470"
        fill="none"
        stroke={palette.caution}
        strokeWidth="4"
        strokeLinecap="round"
        markerEnd="url(#arrowhead)"
      />
      <text
        x="1012"
        y="400"
        fill={palette.caution}
        fontFamily={typography.body}
        fontSize="14"
        fontWeight="700"
      >
        low confidence
      </text>

      <path
        d="M1160 384 C1160 446, 940 436, 940 244"
        fill="none"
        stroke={palette.danger}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray="10 10"
        markerEnd="url(#alertArrow)"
      />
      <text
        x="979"
        y="426"
        fill={palette.danger}
        fontFamily={typography.body}
        fontSize="14"
        fontWeight="700"
      >
        conflict -> escalate
      </text>

      <rect
        x="1328"
        y="390"
        width="214"
        height="86"
        rx="20"
        fill={palette.panelAlert}
        stroke={palette.line}
        strokeWidth="2"
      />
      <text
        x="1348"
        y="424"
        fill={palette.ink}
        fontFamily={typography.display}
        fontSize="22"
        fontWeight="700"
      >
        Human Review
      </text>
      <text
        x="1348"
        y="452"
        fill={palette.muted}
        fontFamily={typography.body}
        fontSize="15"
      >
        Triggered on weak or
      </text>
      <text
        x="1348"
        y="472"
        fill={palette.muted}
        fontFamily={typography.body}
        fontSize="15"
      >
        contradictory evidence
      </text>

      <path
        d="M980 258 C1206 258, 1280 390, 1328 430"
        fill="none"
        stroke={palette.danger}
        strokeWidth="4"
        strokeLinecap="round"
        markerEnd="url(#alertArrow)"
      />

      <text
        x="72"
        y="868"
        fill={palette.muted}
        fontFamily={typography.mono}
        fontSize="14"
      >
        pl = probability_of_loss | confidence is computed in the controller, not in predict_loss
      </text>
    </svg>
  );
}
