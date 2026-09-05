// Press → Handstand · self-evolving coach (TypeScript client)
//
// Uses the Rust engine compiled to WebAssembly when available (static GitHub
// Pages — no server). If the WASM module isn't present, it falls back to the
// Rust HTTP API (running the server locally with `./target/release/p2h-engine`).

interface CapMeta { name: string; hint: string; }
interface Meta { capacities: CapMeta[]; skills: {name:string}[]; weeks:number; threshold:number; goal:number; goalName:string; }
interface StateResp { gen:number; best_fitness:number; population:number; profiles?:number; running:boolean; history:number[]; }
interface RecommendResp {
  caps:number[]; startAttainment:number; finalAttainment:number;
  weeksToPress:number; injuries:number; source:"evolved"|"baseline";
  evoAttainment:number; baseAttainment:number; gen:number;
  trace:number[]; blocks:{skill:number;name:string;weekStart:number;weeks:number}[]; ranking:{rank:number;skill:number;name:string;focus:number}[];
}

const $ = (id:string)=>document.getElementById(id) as HTMLElement;
let meta: Meta | null = null;
let curCaps: number[] = [];
let evolTimer: number | undefined;

// ---------- backend (wasm engine or HTTP API) ----------
const hasWasm = typeof WebAssembly !== "undefined";
let useWasm = false;

interface Backend {
  load(): Promise<void>;
  meta(): Promise<Meta>;
  state(): Promise<StateResp>;
  recommend(caps:number[]): Promise<RecommendResp>;
  warm(n:number): void;
}

async function makeWasmBackend(): Promise<Backend | null> {
  if (!hasWasm) return null;
  try {
    const resp = await fetch("p2h.wasm");
    if (!resp.ok) return null;
    // Use arrayBuffer + instantiate (robust on Pages: handles any MIME/encoding,
    // unlike instantiateStreaming).
    const bytes = await resp.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const e = instance.exports as Record<string, any>;
    const read = (fn:()=>number):string => {
      const ptr = fn();
      const len = e.p2h_out_len();
      return new TextDecoder().decode(new Uint8Array(e.memory.buffer, ptr, len));
    };
    useWasm = true;
    return {
      load: async () => { e.p2h_init(1337n); e.p2h_run(150n); },  // u64 ⇒ BigInt
      meta: async () => JSON.parse(read(()=>e.p2h_meta())),
      state: async () => JSON.parse(read(()=>e.p2h_state())),
      recommend: async (caps:number[]) => {
        const n = 8;
        const ptr = e.p2h_alloc_f32(n);                 // pointer/len are i32 ⇒ Number fine
        new Float32Array(e.memory.buffer, ptr, n).set(caps);
        return JSON.parse(read(()=>e.p2h_recommend(ptr, n)));
      },
      warm: (n:number) => e.p2h_run(BigInt(n)),         // u64 ⇒ BigInt
    };
  } catch (err) { console.error("wasm backend failed, falling back to HTTP:", err); useWasm = false; return null; }
}

function makeHttpBackend(): Backend {
  useWasm = false;
  return {
    load: async () => { try { await fetch("/api/meta"); } catch (_) {} },
    meta: async () => await (await fetch("/api/meta")).json(),
    state: async () => await (await fetch("/api/evolution")).json(),
    recommend: async (caps:number[]) => await (await fetch("/api/recommend", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({caps}) })).json(),
    warm: () => {},
  };
}
let backend: Backend;

async function fetchState(): Promise<StateResp> {
  return await backend.state();
}

function el<T extends HTMLElement=HTMLElement>(tag:string,cls?:string):T{const e=document.createElement(tag) as T;if(cls)e.className=cls;return e;}
function fmtPct(x:number){return Math.round(x*100)+"%";}

// ---------- canvas line chart ----------
function drawLine(cv:HTMLCanvasElement,data:number[],opts:{min?:number;max?:number;color:string}){
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth||520,h=cv.clientHeight||180;
  cv.width=w*dpr;cv.height=h*dpr;
  const ctx=cv.getContext("2d")!;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const padL=8,padR=8,padT=10,padB=12,iw=w-padL-padR,ih=h-padT-padB;
  const n=data.length;if(n<2)return;
  let mn=opts.min??Math.min(...data),mx=opts.max??Math.max(...data);
  if(mx-mn<1e-6){mx=mn+1;}
  ctx.strokeStyle="#223045";ctx.lineWidth=1;
  for(let g=0;g<=3;g++){const y=padT+(ih*g)/3;ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(padL+iw,y);ctx.stroke();}
  const X=(i:number)=>padL+(i/(n-1))*iw,Y=(v:number)=>padT+ih*(1-(v-mn)/(mx-mn));
  const grad=ctx.createLinearGradient(0,padT,0,padT+ih);
  grad.addColorStop(0,"rgba(79,209,197,.30)");grad.addColorStop(1,"rgba(79,209,197,0)");
  ctx.beginPath();ctx.moveTo(X(0),Y(data[0]));
  for(let i=1;i<n;i++)ctx.lineTo(X(i),Y(data[i]));
  ctx.lineTo(X(n-1),padT+ih);ctx.lineTo(X(0),padT+ih);ctx.closePath();
  ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();ctx.moveTo(X(0),Y(data[0]));
  for(let i=1;i<n;i++)ctx.lineTo(X(i),Y(data[i]));
  ctx.strokeStyle=opts.color;ctx.lineWidth=2;ctx.lineJoin="round";ctx.stroke();
  ctx.fillStyle="#8aa0b8";ctx.font="10px sans-serif";
  ctx.fillText((mx>=100?mx.toFixed(0):mx.toFixed(1)).toString(),padL+2,padT+9);
  ctx.fillText(mn.toFixed(1).toString(),padL+2,padT+ih-2);
}

// ---------- nav ----------
function setupNav(){
  const navEvo=$("navEvo"),navPlan=$("navPlan"),secEvo=$("secEvo"),secPlan=$("secPlan");
  navEvo.onclick=()=>{secEvo.classList.remove("hidden");secPlan.classList.add("hidden");navEvo.classList.add("on");navPlan.classList.remove("on");};
  navPlan.onclick=()=>{secEvo.classList.add("hidden");secPlan.classList.remove("hidden");navPlan.classList.add("on");navEvo.classList.remove("on");};
}

// ---------- evolution ----------
async function pollEvolution(){
  try{
    const d=await fetchState();
    $("sGen").textContent=d.gen.toLocaleString();
    $("sFit").textContent=d.best_fitness.toFixed(1);
    $("sPop").textContent=String(d.population);
    drawLine($("evoChart") as HTMLCanvasElement,d.history.length?d.history:[d.best_fitness],{color:"#4fd1c5"});
  }catch(_){}
}

// ---------- sliders ----------
function buildSliders(){
  const host=$("sliders");host.innerHTML="";
  meta!.capacities.forEach((c,i)=>{
    const row=el("div","sliderrow");
    const top=el("div","top");
    const name=el("div","name");name.textContent=c.name;
    const val=el("div","val");val.textContent=Math.round((curCaps[i]??0.5)*100)+"%";
    top.append(name,val);
    const hint=el("div","hint");hint.textContent=c.hint;
    const input=el("input") as HTMLInputElement;input.type="range";input.min="0";input.max="100";
    input.value=String(Math.round((curCaps[i]??0.5)*100));
    input.style.setProperty("--p",input.value+"%");
    input.oninput=()=>{curCaps[i]=Number(input.value)/100;val.textContent=input.value+"%";input.style.setProperty("--p",input.value+"%");};
    row.append(top,hint,input);host.appendChild(row);
  });
}

// ---------- recommend ----------
let planChart:number[]=[];
async function recommend(){
  $("planHint").textContent="Computing your pathway…";
  const d:RecommendResp = await backend.recommend(curCaps);
  renderPlan(d);
}
function renderPlan(d:RecommendResp){
  const reached=d.weeksToPress>0;
  $("planHint").textContent="The coach ("+(d.source==="evolved"?"evolved neural network":"greedy baseline")+") schedules a "+meta!.weeks+"-week programme. Refine your profile to personalise.";
  $("pWeeks").textContent=reached?d.weeksToPress+" wk":">"+meta!.weeks+" wk";
  $("pFinal").textContent=fmtPct(d.finalAttainment);
  $("pInj").textContent=String(d.injuries);
  const bar=$("pBar") as HTMLElement;bar.style.width=Math.min(100,d.finalAttainment*100)+"%";
  bar.style.background=reached?"linear-gradient(90deg,#3ddc97,#4fd1c5)":"linear-gradient(90deg,#ffb454,#4fd1c5)";
  const src=$("planSrc");src.className="pill "+(d.source==="evolved"?"ok":"start");
  src.textContent=d.source==="evolved"?"evolved NN":"greedy";
  planChart=d.trace;
  drawLine($("planChart") as HTMLCanvasElement,d.trace,{min:0,max:1,color:reached?"#3ddc97":"#ffb454"});
  const bl=$("pBlocks");bl.innerHTML="";
  d.blocks.forEach((b,i)=>{
    const row=el("div","block");
    const phase=el("div","phase");phase.textContent="WK "+(b.weekStart+1)+"–"+(b.weekStart+b.weeks);
    const desc=el("div","desc");desc.textContent=(i+1)+". "+b.name;
    const focus=el("span","focus");focus.textContent=b.weeks+" week"+(b.weeks>1?"s":"")+" primary focus";
    desc.appendChild(focus);row.append(phase,desc);bl.appendChild(row);
  });
  const cb=$("capBars");cb.innerHTML="";
  const last=d.trace[d.trace.length-1]||d.finalAttainment;
  const startAvg=d.startAttainment||0.3;
  meta!.capacities.forEach((c,i)=>{
    const start=d.caps[i]??0.5;
    const growth=(last-startAvg)/Math.max(0.001,1-startAvg);
    const end=Math.min(0.95,start+(1-start)*Math.max(0,growth));
    const row=el("div","caprow");
    const nm=el("div","nm");nm.textContent=c.name;
    const barW=el("div","bar");barW.style.position="relative";
    const a=el("i");a.style.width=(start*100)+"%";a.style.background="#5b6f8a";a.style.position="absolute";a.style.top="0";a.style.left="0";
    const e=el("i");e.style.width=(end*100)+"%";e.style.background="linear-gradient(90deg,#4fd1c5,#7aa2ff)";e.style.height="100%";
    barW.appendChild(a);barW.appendChild(e);
    const lbl=el("div","lbl");lbl.textContent=Math.round(start*100)+"%→"+Math.round(end*100)+"%";
    row.append(nm,barW,lbl);cb.appendChild(row);
  });
  $("pSource").textContent="Coach generation "+d.gen.toLocaleString()+" · evolved NN "+fmtPct(d.evoAttainment)+" · greedy baseline "+fmtPct(d.baseAttainment)+" · press threshold "+fmtPct(meta!.threshold)+(useWasm?" · engine: Rust (WebAssembly)":" · engine: Rust (HTTP server)");
}

// ---------- init ----------
async function init(){
  setupNav();
  curCaps=[0.55,0.5,0.5,0.45,0.5,0.5,0.45,0.5];

  // Choose engine: Rust→WebAssembly in the browser, else the Rust HTTP server.
  try {
    const wasmBackend = await makeWasmBackend();
    backend = wasmBackend ?? makeHttpBackend();
    await backend.load();
    meta = await backend.meta();
  } catch (err) {
    console.error("engine init failed:", err);
    backend = makeHttpBackend();   // last-resort fallback
  }
  if (!meta) {
    meta = { capacities:[], skills:[], weeks:36, threshold:0.8, goal:9, goalName:"Full press to handstand" } as Meta;
  }

  buildSliders();
  pollEvolution();
  evolTimer = window.setInterval(pollEvolution, 900);
  $("btnRecompute").onclick = recommend;

  // In wasm mode the GA runs in-browser; nudge it forward so the fitness curve
  // keeps climbing without freezing the UI (run in small chunks).
  if (useWasm) {
    window.setInterval(() => { try { backend.warm(25); pollEvolution(); } catch(_){} }, 1800);
  }

  try { await recommend(); } catch (err) { console.error("initial plan failed:", err); $("planHint").textContent="Engine failed to produce a plan — see console."; }
}
init();
