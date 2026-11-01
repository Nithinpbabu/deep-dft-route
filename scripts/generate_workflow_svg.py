"""
Generates a modern, enterprise-grade SVG flowchart comparing
the Traditional ASIC Physical Design Flow vs. the DeepDFT-Route Augmented Flow.
"""

from pathlib import Path

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 1620" width="100%" height="100%">
  <defs>
    <!-- Background Gradient -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#090d16" />
      <stop offset="50%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#090d16" />
    </linearGradient>

    <!-- Card Gradients -->
    <linearGradient id="sharedCardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e293b" />
      <stop offset="100%" stop-color="#0f172a" />
    </linearGradient>

    <linearGradient id="tradCardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2d151d" />
      <stop offset="100%" stop-color="#1a0b11" />
    </linearGradient>

    <linearGradient id="aiCardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0d2826" />
      <stop offset="100%" stop-color="#061a1b" />
    </linearGradient>

    <linearGradient id="highlightAIGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#065f46" />
      <stop offset="100%" stop-color="#047857" />
    </linearGradient>

    <!-- Drop Shadows -->
    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="125%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.5" />
    </filter>

    <filter id="glowGreen" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>

    <!-- Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#f87171" />
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#34d399" />
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
    </marker>
  </defs>

  <style>
    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 800; font-size: 26px; fill: #f8fafc; letter-spacing: 0.5px; }
    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; fill: #94a3b8; }
    .col-header { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #f1f5f9; }
    .card-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #94a3b8; }
    .card-desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #cbd5e1; }
    .tag { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 10.5px; }
    .time-badge { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11.5px; }
    .kpi-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; fill: #94a3b8; text-transform: uppercase; }
    .kpi-val { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 800; }
  </style>

  <!-- Deep Canvas Background -->
  <rect width="1240" height="1620" fill="url(#bgGrad)" />

  <!-- Background Decorative Grid Lines -->
  <g opacity="0.05" stroke="#94a3b8" stroke-width="1">
    <line x1="620" y1="0" x2="620" y2="1620" stroke-dasharray="8 8" />
    <line x1="0" y1="520" x2="1240" y2="520" />
    <line x1="0" y1="1130" x2="1240" y2="1130" />
  </g>

  <!-- ==================== HEADER ==================== -->
  <g transform="translate(620, 50)" text-anchor="middle">
    <rect x="-310" y="-12" width="620" height="26" rx="13" fill="#1e293b" stroke="#334155" stroke-width="1" />
    <text y="5" fill="#38bdf8" font-family="sans-serif" font-size="11" font-weight="700" letter-spacing="1.2">ENTERPRISE VLSI &amp; ASIC SEMICONDUCTOR WORKFLOW</text>
    
    <text y="48" class="title">Physical Design Flow with DeepDFT-Route Surrogate</text>
    <text y="74" class="subtitle">Side-by-Side Comparison: Traditional PnR Routing Iteration vs. Shift-Left Neural Hotspot Prediction</text>
  </g>

  <!-- ==================== 1. SHARED UPSTREAM STAGES ==================== -->

  <!-- Stage 1: RTL Freeze -->
  <g transform="translate(370, 150)">
    <rect width="500" height="68" rx="10" fill="url(#sharedCardGrad)" stroke="#334155" stroke-width="1.5" filter="url(#cardShadow)" />
    <circle cx="34" cy="34" r="16" fill="#1e3a8a" />
    <text x="34" y="39" text-anchor="middle" fill="#60a5fa" font-weight="bold" font-size="12">1</text>
    <text x="65" y="30" class="card-title">RTL Design Freeze &amp; Elaboration</text>
    <text x="65" y="48" class="card-desc">SystemVerilog / Verilog RTL defining core logic, ALUs, memory cuts, and buses.</text>
    <!-- Tag -->
    <rect x="385" y="16" width="98" height="20" rx="5" fill="#1e293b" stroke="#475569" stroke-width="1" />
    <text x="434" y="30" text-anchor="middle" fill="#94a3b8" class="tag">Frontend Team</text>
  </g>

  <!-- Down Arrow 1 -->
  <path d="M 620 218 L 620 244" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Stage 2: Synthesis -->
  <g transform="translate(370, 246)">
    <rect width="500" height="74" rx="10" fill="url(#sharedCardGrad)" stroke="#334155" stroke-width="1.5" filter="url(#cardShadow)" />
    <circle cx="34" cy="37" r="16" fill="#1e3a8a" />
    <text x="34" y="42" text-anchor="middle" fill="#60a5fa" font-weight="bold" font-size="12">2</text>
    <text x="65" y="29" class="card-title">Logic Synthesis &amp; Target Cell Mapping</text>
    <text x="65" y="46" class="card-desc">Maps RTL to standard cells (.lib). Optimizes timing constraints (.sdc).</text>
    <!-- Tool Pills -->
    <rect x="65" y="52" width="135" height="16" rx="4" fill="#0f172a" stroke="#38bdf8" stroke-width="1" />
    <text x="132" y="64" text-anchor="middle" fill="#38bdf8" font-size="9.5" font-weight="600">Synopsys Design Compiler</text>
    <rect x="207" y="52" width="90" height="16" rx="4" fill="#0f172a" stroke="#64748b" stroke-width="1" />
    <text x="252" y="64" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-weight="600">Cadence Genus</text>
  </g>

  <!-- Down Arrow 2 -->
  <path d="M 620 320 L 620 346" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Stage 3: Tessent DFT Insertion -->
  <g transform="translate(370, 348)">
    <rect width="500" height="82" rx="10" fill="url(#sharedCardGrad)" stroke="#2563eb" stroke-width="1.8" filter="url(#cardShadow)" />
    <circle cx="34" cy="41" r="16" fill="#1d4ed8" />
    <text x="34" y="46" text-anchor="middle" fill="#bfdbfe" font-weight="bold" font-size="12">3</text>
    <text x="65" y="28" class="card-title">DFT Architecture &amp; Scan Insertion</text>
    <text x="65" y="46" class="card-desc">SDFF replacement, EDT compression blocks, Scan-Enable (SE) trees &amp; chains.</text>
    <!-- Outputs & Tool -->
    <rect x="65" y="57" width="140" height="17" rx="4" fill="#172554" stroke="#60a5fa" stroke-width="1" />
    <text x="135" y="69" text-anchor="middle" fill="#93c5fd" font-size="9.5" font-weight="700">Siemens Tessent Scan / EDT</text>
    <text x="215" y="70" class="card-sub" fill="#94a3b8">Exports: ScanDEF, TSDB, Netlist (.v)</text>
  </g>

  <!-- Down Arrow 3 -->
  <path d="M 620 430 L 620 456" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />

  <!-- Stage 4: Early Physical Design (Placement) -->
  <g transform="translate(370, 458)">
    <rect width="500" height="78" rx="10" fill="url(#sharedCardGrad)" stroke="#334155" stroke-width="1.5" filter="url(#cardShadow)" />
    <circle cx="34" cy="39" r="16" fill="#1e3a8a" />
    <text x="34" y="44" text-anchor="middle" fill="#60a5fa" font-weight="bold" font-size="12">4</text>
    <text x="65" y="28" class="card-title">Early Physical Design (Floorplan &amp; Global Placement)</text>
    <text x="65" y="45" class="card-desc">Die boundary, power grid (VDD/VSS), macro floorplanning, standard cell legal placement.</text>
    <rect x="65" y="55" width="135" height="16" rx="4" fill="#0f172a" stroke="#38bdf8" stroke-width="1" />
    <text x="132" y="67" text-anchor="middle" fill="#38bdf8" font-size="9.5" font-weight="600">Synopsys IC Compiler II</text>
    <rect x="207" y="55" width="105" height="16" rx="4" fill="#0f172a" stroke="#64748b" stroke-width="1" />
    <text x="259" y="67" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-weight="600">Cadence Innovus</text>
  </g>

  <!-- ==================== THE FORK (SPLIT PATH) ==================== -->
  <path d="M 620 536 L 620 565" stroke="#64748b" stroke-width="2" />
  <path d="M 620 565 L 305 565 L 305 600" stroke="#f87171" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#arrowRed)" />
  <path d="M 620 565 L 935 565 L 935 600" stroke="#34d399" stroke-width="2.5" marker-end="url(#arrowGreen)" />

  <!-- ==================== LEFT COLUMN: TRADITIONAL FLOW ==================== -->
  
  <!-- Column Header Left -->
  <g transform="translate(70, 600)">
    <rect width="470" height="36" rx="6" fill="#450a0a" stroke="#b91c1c" stroke-width="1" />
    <text x="235" y="23" text-anchor="middle" fill="#fca5a5" class="col-header">Traditional Flow (Status Quo)</text>
  </g>

  <!-- Step L1: Run Full Detailed Route Directly -->
  <g transform="translate(70, 650)">
    <rect width="470" height="96" rx="10" fill="url(#tradCardGrad)" stroke="#7f1d1d" stroke-width="1.5" filter="url(#cardShadow)" />
    <text x="20" y="26" class="card-title" fill="#fecaca">Launch Full Detailed Route on Server Cluster</text>
    <text x="20" y="47" class="card-desc">Proceeds blindly into TritonRoute / ZRoute / NanoRoute.</text>
    <text x="20" y="64" class="card-desc">Routes all signal nets, clock trees, and long scan stitch wires.</text>
    
    <!-- Time Warning Badge -->
    <rect x="20" y="72" width="190" height="18" rx="4" fill="#7f1d1d" />
    <text x="115" y="85" text-anchor="middle" fill="#fee2e2" class="time-badge">&#x23F3; Takes 12 to 48+ Hours</text>
    <text x="220" y="85" fill="#fca5a5" font-size="11">Heavy compute farm load</text>
  </g>

  <!-- Arrow L1 -> L2 -->
  <path d="M 305 746 L 305 776" stroke="#f87171" stroke-width="2" marker-end="url(#arrowRed)" />

  <!-- Step L2: Late Congestion Blowout -->
  <g transform="translate(70, 778)">
    <rect width="470" height="110" rx="10" fill="url(#tradCardGrad)" stroke="#dc2626" stroke-width="2" filter="url(#cardShadow)" />
    <circle cx="34" cy="30" r="12" fill="#ef4444" />
    <text x="34" y="34" text-anchor="middle" fill="#ffffff" font-weight="bold" font-size="13">&#x2715;</text>
    <text x="55" y="32" class="card-title" fill="#f87171">Late Congestion &amp; DRC Hotspot Explosion</text>
    <text x="20" y="60" class="card-desc">Detailed router fails to converge due to scan chains crossing</text>
    <text x="20" y="76" class="card-desc">narrow macro channels and scan-enable high fanout stress.</text>
    <rect x="20" y="86" width="220" height="16" rx="4" fill="#450a0a" stroke="#991b1b" stroke-width="1" />
    <text x="130" y="98" text-anchor="middle" fill="#fca5a5" font-size="10" font-weight="600">Discovered Days Later in Post-Route</text>
  </g>

  <!-- Painful Red Loop Back Up -->
  <path d="M 70 833 C 15 833 15 400 355 400" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="6 6" marker-end="url(#arrowRed)" />
  <g transform="translate(25, 600)">
    <rect x="-10" y="-12" width="135" height="42" rx="6" fill="#18080a" stroke="#ef4444" stroke-width="1" />
    <text x="57" y="4" text-anchor="middle" fill="#f87171" font-size="10.5" font-weight="700">PAINFUL DELAY</text>
    <text x="57" y="20" text-anchor="middle" fill="#fca5a5" font-size="10">3 to 7 Days Lost / Loop</text>
  </g>

  <!-- Arrow L2 -> L3 -->
  <path d="M 305 888 L 305 918" stroke="#f87171" stroke-width="2" marker-end="url(#arrowRed)" />

  <!-- Step L3: Expensive Re-iterations -->
  <g transform="translate(70, 920)">
    <rect width="470" height="88" rx="10" fill="url(#tradCardGrad)" stroke="#7f1d1d" stroke-width="1.5" filter="url(#cardShadow)" />
    <text x="20" y="26" class="card-title" fill="#fca5a5">Repeated PnR Trial &amp; Error Cycles</text>
    <text x="20" y="46" class="card-desc">Manual bug reports between PD and DFT teams. Multiple weekend runs.</text>
    <text x="20" y="64" class="card-desc">Risk of tapeout schedule slip; massive compute license consumption.</text>
    <rect x="20" y="70" width="130" height="14" rx="3" fill="#450a0a" />
    <text x="85" y="81" text-anchor="middle" fill="#ef4444" font-size="9" font-weight="700">HIGH SCHEDULE RISK</text>
  </g>

  <!-- ==================== RIGHT COLUMN: DEEPDFT-ROUTE ==================== -->

  <!-- Column Header Right -->
  <g transform="translate(700, 600)">
    <rect width="470" height="36" rx="6" fill="#064e3b" stroke="#059669" stroke-width="1.5" />
    <text x="235" y="23" text-anchor="middle" fill="#6ee7b7" class="col-header">DeepDFT-Route Flow (Shift-Left)</text>
  </g>

  <!-- Step R1: Pre-Route Feature Extraction -->
  <g transform="translate(700, 650)">
    <rect width="470" height="96" rx="10" fill="url(#aiCardGrad)" stroke="#065f46" stroke-width="1.5" filter="url(#cardShadow)" />
    <text x="20" y="26" class="card-title" fill="#a7f3d0">Pre-Route Feature &amp; Graph Extraction</text>
    <text x="20" y="46" class="card-desc">Extracts 7-channel 2D spatial layout tensor from early placement DEF.</text>
    <text x="20" y="63" class="card-desc">Extracts circuit netlist &amp; Tessent scan chain connectivity graph.</text>
    <rect x="20" y="72" width="150" height="18" rx="4" fill="#064e3b" stroke="#10b981" stroke-width="1" />
    <text x="95" y="85" text-anchor="middle" fill="#34d399" class="time-badge">&#x23F1; Run Time: ~2 Mins</text>
    <text x="180" y="85" fill="#a7f3d0" font-size="11">Pre-Route Compatible</text>
  </g>

  <!-- Arrow R1 -> R2 -->
  <path d="M 935 746 L 935 776" stroke="#34d399" stroke-width="2" marker-end="url(#arrowGreen)" />

  <!-- Step R2: DeepDFT-Route Surrogate Inference (HERO BOX) -->
  <g transform="translate(700, 778)">
    <rect width="470" height="110" rx="10" fill="url(#aiCardGrad)" stroke="#10b981" stroke-width="2.5" filter="url(#cardShadow)" />
    <!-- Glowing AI Badge -->
    <rect x="20" y="16" width="160" height="22" rx="11" fill="url(#highlightAIGrad)" filter="url(#glowGreen)" />
    <text x="100" y="31" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="800" letter-spacing="0.5">&#x26A1; DEEPDFT-ROUTE</text>
    
    <text x="195" y="32" class="card-title" fill="#6ee7b7">Multimodal CNN + GNN Inference</text>
    <text x="20" y="58" class="card-desc">Fuses 2D spatial layout receptive fields with GraphSAGE circuit topology.</text>
    <text x="20" y="74" class="card-desc">Predicts continuous wirelength density &amp; Top-5% congestion alert mask.</text>
    
    <rect x="20" y="83" width="175" height="18" rx="4" fill="#064e3b" stroke="#34d399" stroke-width="1" />
    <text x="107" y="96" text-anchor="middle" fill="#6ee7b7" class="time-badge">&#x26A1; Inference: &lt; 1 Second</text>
    <text x="205" y="96" fill="#a7f3d0" font-size="10.5" font-weight="600">92.1% Critical Hotspot Recall</text>
  </g>

  <!-- Rapid Shift-Left Green Loop -->
  <path d="M 1170 833 C 1225 833 1225 400 885 400" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="6 6" marker-end="url(#arrowGreen)" />
  <g transform="translate(1100, 600)">
    <rect x="-10" y="-12" width="145" height="42" rx="6" fill="#042018" stroke="#10b981" stroke-width="1.2" />
    <text x="62" y="4" text-anchor="middle" fill="#34d399" font-size="10.5" font-weight="700">SHIFT-LEFT LOOP</text>
    <text x="62" y="20" text-anchor="middle" fill="#a7f3d0" font-size="10">30 - 45 Min Resolution</text>
  </g>

  <!-- Arrow R2 -> R3 -->
  <path d="M 935 888 L 935 918" stroke="#34d399" stroke-width="2" marker-end="url(#arrowGreen)" />

  <!-- Step R3: Confident 1-Pass Route -->
  <g transform="translate(700, 920)">
    <rect width="470" height="88" rx="10" fill="url(#aiCardGrad)" stroke="#065f46" stroke-width="1.5" filter="url(#cardShadow)" />
    <text x="20" y="26" class="card-title" fill="#a7f3d0">1-Pass High-Confidence Detailed Routing</text>
    <text x="20" y="46" class="card-desc">Scan re-stitching or Tessent pipeline registers resolved hotspots earlier.</text>
    <text x="20" y="64" class="card-desc">Detailed route converges on first pass with zero DRC / congestion surprises.</text>
    <rect x="20" y="70" width="150" height="14" rx="3" fill="#064e3b" />
    <text x="95" y="81" text-anchor="middle" fill="#34d399" font-size="9" font-weight="700">&#x2713; FIRST-PASS SUCCESS</text>
  </g>

  <!-- ==================== MERGE DOWNSTREAM ==================== -->
  <path d="M 305 1008 L 305 1045 L 620 1045" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 4" />
  <path d="M 935 1008 L 935 1045 L 620 1045" stroke="#34d399" stroke-width="2" />
  <path d="M 620 1045 L 620 1070" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowBlue)" />

  <!-- ==================== 5. FINAL SIGN-OFF & TAPEOUT ==================== -->

  <!-- Final Sign-off Stage -->
  <g transform="translate(320, 1072)">
    <rect width="600" height="88" rx="10" fill="url(#sharedCardGrad)" stroke="#38bdf8" stroke-width="2" filter="url(#cardShadow)" />
    <circle cx="38" cy="44" r="18" fill="#0369a1" />
    <text x="38" y="50" text-anchor="middle" fill="#ffffff" font-weight="bold" font-size="14">5</text>
    <text x="75" y="30" class="card-title">Full Sign-off &amp; Physical Verification</text>
    <text x="75" y="48" class="card-desc">Timing closure, signal integrity, manufacturing ATPG pattern generation, and DRC/LVS.</text>
    <!-- Tool Pills -->
    <rect x="75" y="58" width="135" height="18" rx="4" fill="#0f172a" stroke="#38bdf8" stroke-width="1" />
    <text x="142" y="71" text-anchor="middle" fill="#38bdf8" font-size="10" font-weight="700">Synopsys PrimeTime (STA)</text>
    <rect x="220" y="58" width="145" height="18" rx="4" fill="#0f172a" stroke="#60a5fa" stroke-width="1" />
    <text x="292" y="71" text-anchor="middle" fill="#93c5fd" font-size="10" font-weight="700">Tessent TestKompress (ATPG)</text>
    <rect x="375" y="58" width="135" height="18" rx="4" fill="#0f172a" stroke="#64748b" stroke-width="1" />
    <text x="442" y="71" text-anchor="middle" fill="#cbd5e1" font-size="10" font-weight="700">Siemens Calibre (DRC)</text>
  </g>

  <!-- Down Arrow to Tapeout -->
  <path d="M 620 1160 L 620 1186" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowBlue)" />

  <!-- Tapeout GDSII Box -->
  <g transform="translate(420, 1188)">
    <rect width="400" height="54" rx="8" fill="#1e293b" stroke="#22c55e" stroke-width="2" />
    <text x="200" y="24" text-anchor="middle" fill="#4ade80" font-weight="800" font-size="14" letter-spacing="1">&#x2714; TAPE-OUT TO FOUNDRY</text>
    <text x="200" y="42" text-anchor="middle" fill="#94a3b8" font-size="11.5">Clean GDSII / OASIS Mask Generation (TSMC / Intel / Samsung)</text>
  </g>

  <!-- ==================== 6. EXECUTIVE COMPARISON KPI SUMMARY ==================== -->
  
  <g transform="translate(70, 1280)">
    <!-- KPI Container -->
    <rect width="1100" height="140" rx="12" fill="#0b1329" stroke="#1e293b" stroke-width="1.5" />
    
    <!-- Title -->
    <text x="550" y="32" text-anchor="middle" fill="#f8fafc" font-family="sans-serif" font-size="14" font-weight="700" letter-spacing="0.5">EXECUTIVE SUMMARY: SHIFT-LEFT IMPACT ON CHIP DEVELOPMENT</text>
    <line x1="40" y1="46" x2="1060" y2="46" stroke="#1e293b" stroke-width="1" />

    <!-- KPI 1: TAT -->
    <g transform="translate(60, 62)">
      <text x="0" y="16" class="kpi-title">Turnaround Time (TAT)</text>
      <text x="0" y="42" class="kpi-val" fill="#f87171">3 to 7 Days</text>
      <text x="125" y="42" font-size="16" fill="#64748b">&#x2192;</text>
      <text x="155" y="42" class="kpi-val" fill="#34d399">30 - 45 Mins</text>
      <text x="0" y="60" font-size="11" fill="#38bdf8" font-weight="600">~90% Reduction in feedback latency</text>
    </g>

    <!-- Divider 1 -->
    <line x1="390" y1="58" x2="390" y2="128" stroke="#1e293b" stroke-width="1" />

    <!-- KPI 2: Compute Cost -->
    <g transform="translate(430, 62)">
      <text x="0" y="16" class="kpi-title">Compute Cluster &amp; EDA Licenses</text>
      <text x="0" y="42" class="kpi-val" fill="#f87171">Heavy Cluster</text>
      <text x="145" y="42" font-size="16" fill="#64748b">&#x2192;</text>
      <text x="175" y="42" class="kpi-val" fill="#34d399">&lt; 1s GPU Pass</text>
      <text x="0" y="60" font-size="11" fill="#38bdf8" font-weight="600">Eliminates wasted 48h router runs</text>
    </g>

    <!-- Divider 2 -->
    <line x1="770" y1="58" x2="770" y2="128" stroke="#1e293b" stroke-width="1" />

    <!-- KPI 3: First-Pass Routability -->
    <g transform="translate(810, 62)">
      <text x="0" y="16" class="kpi-title">First-Pass Route Success</text>
      <text x="0" y="42" class="kpi-val" fill="#f87171">&lt; 40% (DFT Risk)</text>
      <text x="165" y="42" font-size="16" fill="#64748b">&#x2192;</text>
      <text x="195" y="42" class="kpi-val" fill="#34d399">&gt; 95% Confident</text>
      <text x="0" y="60" font-size="11" fill="#38bdf8" font-weight="600">92.1% Recall on unseen test circuits</text>
    </g>
  </g>

</svg>
"""

output_path = Path("docs/assets/enterprise_workflow_comparison.svg")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(svg_content, encoding="utf-8")
print(f"Generated clean SVG at: {output_path.resolve()}")
