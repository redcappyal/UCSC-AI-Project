/* Archived 2026-07-29. Excised verbatim from index.html — see README.md.
   Nothing here is loaded, bundled, linted, or tested. `$`, `S`, `ctx`, `canvas`,
   `vid`, `showError`, `redrawTrackCanvas` and friends are index.html globals;
   this file will not run standalone and is not meant to.

   Restoring this means putting the pieces back where the "Called from" notes
   say, not importing this file. */

/* ---------- state fields removed from the `S` literal ---------- */
const ARCHIVED_STATE_FIELDS = {
  corrections: {},          // frame -> saved v2 correction entry for the run
  corrTarget: null,         // {frame, predicted, draft, pickingFrame} — the hit being corrected
  ballByFrame: null,        // Map(frame -> {x,y}) from ball_coordinates.csv (video px)
};

/* ---------- call sites that were deleted from surviving functions ----------

   1. After a track run completes (~index.html:5398) and again on session
      restore (~index.html:7612), both immediately after the run landed:
          loadCorrections(data.run_id);
          loadBallPositions(data.run_id);

   2. clearJudgeResult() ended with:
          hideCorrPanel();

   3. renderEventCard(hit) ended with:
          // Non-wall hits carry no judge context; seed the dot from the impact fit
          // when the pipeline produced one, else the nearest detected ball center.
          const ball = hit.impact ? {x:hit.impact.x, y:hit.impact.y} : ballAtFrame(hit.frame);
          setCorrTarget(hit.frame, { type:hit.event_type, call:null, source:null,
                                     margin_px:null, ball });

   4. renderJudgeResult(data, selectedHit) opened with:
          setCorrTarget(data.frame, {
            type: (hit && hit.event_type) || 'wall',
            call: data.call, source: data.source,
            margin_px: data.margin_px, ball: data.ball,
          });

   5. scheduleAutoJudge() guarded against clobbering a correction mid-scrub:
          if(S.corrTarget && S.corrTarget.pickingFrame) return;
      and its doc comment carried the clause "while picking a corrected bounce
      frame (that scrub must not clobber the correction target)".

   6. judgeFrameRequest(frame, quiet) opened the Review pane on an explicit judge:
          if(!quiet) setCallTab('review');    // the verdict lands in the Review pane

   7. redrawTrackCanvas() ended with:
          drawCorrectionDot();

   `hitAtFrame()` was defined in this block but is NOT archived — the judge flow
   uses it independently, so it stayed in index.html. */

/* ---------- bounce corrections (the data flywheel's front door) ----------
   Every judged hit gets a correction panel: hit type (supervises the event
   engine + false positives), ball position via draggable dot (RF-DETR
   retraining data), and bounce timing (supervises impact estimation).
   IN/OUT stays, but only for front-wall hits — it's derived geometry. */
const CORR_ALL_BTNS = ['corrTypeSel', 'corrFrameThis', 'corrFrameOther'];

/* Review / Challenge tab switch. Judging always lands on Review; Challenge
   holds the correction controls for the currently judged hit. */
function setCallTab(tab){
  $('tabReview').classList.toggle('active', tab === 'review');
  $('tabChallenge').classList.toggle('active', tab === 'challenge');
  $('reviewTab').classList.toggle('hidden', tab !== 'review');
  $('challengeTab').classList.toggle('hidden', tab !== 'challenge');
}
$('tabReview').onclick = () => setCallTab('review');
$('tabChallenge').onclick = () => setCallTab('challenge');

async function loadCorrections(runId){
  S.corrections = {};
  try{
    const response = await fetch(`/api/runs/${runId}/corrections`);
    if(!response.ok) return;
    const data = await response.json();
    for(const c of data.corrections || []) S.corrections[c.frame] = c;
  }catch(_){ /* offline or malformed file — start empty */ }
}

async function loadBallPositions(runId){
  S.ballByFrame = null;
  try{
    const response = await fetch(`/api/runs/${runId}/ball_coordinates.csv`);
    if(!response.ok) return;
    const lines = (await response.text()).trim().split('\n');
    const cols = lines[0].split(',');
    const fi = cols.indexOf('source_frame'), di = cols.indexOf('detected');
    const xi = cols.indexOf('x_center'), yi = cols.indexOf('y_center');
    if(fi < 0 || di < 0 || xi < 0 || yi < 0) return;
    const map = new Map();
    for(const line of lines.slice(1)){
      const row = line.split(',');
      if(row[di].trim().toLowerCase() !== 'true') continue;
      map.set(parseInt(row[fi], 10), {x:+row[xi], y:+row[yi]});
    }
    S.ballByFrame = map;
  }catch(_){ /* dot seeding degrades to null; drag still works from a tap */ }
}

function ballAtFrame(frame){
  if(!S.ballByFrame) return null;
  const exact = S.ballByFrame.get(frame);
  if(exact) return {...exact};
  const radius = 2 * ((S.run && S.run.frame_stride) || 1);
  let best = null, bestDist = Infinity;
  for(const [f, p] of S.ballByFrame){
    const d = Math.abs(f - frame);
    if(d <= radius && d < bestDist){ best = p; bestDist = d; }
  }
  return best ? {...best} : null;
}

function setCorrTarget(frame, predicted){
  const saved = (S.corrections || {})[frame];
  const draft = saved
    ? JSON.parse(JSON.stringify(saved.corrected))
    : { type: predicted.type || 'wall',
        call: predicted.call === 'IN' || predicted.call === 'OUT' ? predicted.call : null,
        ball: predicted.ball ? {...predicted.ball} : null,
        frame_is_bounce: true, frame: null };
  if(draft.type === 'wall' && !draft.call) draft.call = 'IN';
  S.corrTarget = { frame, predicted, draft, pickingFrame:false };
  renderCorrPanel();
  redrawTrackCanvas();
}

function hideCorrPanel(){
  $('corrPanel').classList.add('hidden');
  $('corrStatus').classList.add('hidden');
  $('corrEmpty').classList.remove('hidden');
  if(S.corrTarget){ S.corrTarget = null; redrawTrackCanvas(); }
}

function renderCorrPanel(){
  const target = S.corrTarget;
  if(!target){ hideCorrPanel(); return; }
  const saved = (S.corrections || {})[target.frame];
  const draft = target.draft;
  $('corrTypeSel').value = draft.type === 'wall'
    ? `wall_${draft.call === 'OUT' ? 'OUT' : 'IN'}` : draft.type;
  $('corrFrameRow').classList.toggle('hidden', draft.type === 'none');
  $('corrFrameThis').classList.toggle('active', !!saved && draft.frame_is_bounce === true);
  $('corrFrameOther').classList.toggle('active', !!saved && draft.frame_is_bounce === false);
  $('corrFrameOther').textContent = target.pickingFrame ? 'Use this frame'
    : draft.frame_is_bounce === false && draft.frame != null
      ? `Frame ${draft.frame}` : 'Not bounce';
  $('corrPanel').classList.remove('hidden');
  $('corrEmpty').classList.add('hidden');
  const statusText = target.pickingFrame
    ? 'Scrub to the true bounce frame, then tap Use this frame.'
    : saved ? savedStatusText(saved) : '';
  $('corrStatus').textContent = statusText;
  $('corrStatus').classList.toggle('hidden', !statusText);
}

function savedStatusText(saved){
  const c = saved.corrected;
  if(c.type === 'none') return 'Recorded as not a hit — saved to this run’s eval set.';
  const typeName = { wall:'Front wall', side_wall:'Side wall', floor:'Floor', racket:'Racket' }[c.type] || c.type;
  const call = c.call ? ` ${c.call}` : '';
  const at = c.ball ? ` at (${c.ball.x.toFixed(0)}, ${c.ball.y.toFixed(0)})` : '';
  const when = c.frame_is_bounce === false && c.frame != null ? `, frame ${c.frame}` : '';
  return `Recorded: ${typeName}${call}${at}${when} — saved to this run’s eval set.`;
}

async function sendCorrection(corrected){
  const target = S.corrTarget;
  if(!target || !S.run) return;
  for(const id of CORR_ALL_BTNS) $(id).disabled = true;
  try{
    const response = await fetch(`/api/runs/${S.run.run_id}/corrections`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ frame: target.frame, corrected,
                            predicted: target.predicted }),
    });
    const result = await response.json();
    if(!result.ok) throw new Error(result.error || 'Saving the correction failed.');
    if(corrected === null){
      delete S.corrections[target.frame];
      // Reset the draft to the model's view so the panel reads as untouched.
      setCorrTarget(target.frame, target.predicted);
      return;
    }
    S.corrections[target.frame] = result.correction;
    target.draft = JSON.parse(JSON.stringify(result.correction.corrected));
  }catch(error){
    showError(error.message);
  }finally{
    for(const id of CORR_ALL_BTNS) $(id).disabled = false;
    if(S.corrTarget) renderCorrPanel();
    redrawTrackCanvas();
  }
}

function postDraft(){
  const target = S.corrTarget;
  if(!target) return;
  const d = target.draft;
  const corrected = d.type === 'none'
    ? { type:'none', call:null, ball:null, frame_is_bounce:null, frame:null }
    : { type:d.type, call:d.type === 'wall' ? d.call : null,
        ball:d.ball, frame_is_bounce:d.frame_is_bounce,
        frame:d.frame_is_bounce ? null : d.frame };
  if(corrected.type !== 'none' && !corrected.ball){
    showError('No ball position for this frame — drag the dot onto the ball first.');
    return;
  }
  sendCorrection(corrected);
}

$('corrTypeSel').onchange = () => {
  const target = S.corrTarget;
  if(!target){ return; }
  const value = $('corrTypeSel').value;
  const draft = target.draft;
  if(value === 'wall_IN' || value === 'wall_OUT'){
    draft.type = 'wall';
    draft.call = value === 'wall_OUT' ? 'OUT' : 'IN';
  } else {
    draft.type = value;
    draft.call = null;
  }
  if(draft.type === 'none'){ draft.ball = null; draft.frame_is_bounce = null; draft.frame = null; }
  else {
    if(!draft.ball) draft.ball = target.predicted.ball ? {...target.predicted.ball}
                                                       : ballAtFrame(target.frame);
    if(typeof draft.frame_is_bounce !== 'boolean'){ draft.frame_is_bounce = true; draft.frame = null; }
  }
  target.pickingFrame = false;
  postDraft();
};
$('corrFrameThis').onclick = () => {
  if(!S.corrTarget) return;
  S.corrTarget.draft.frame_is_bounce = true;
  S.corrTarget.draft.frame = null;
  S.corrTarget.pickingFrame = false;
  postDraft();
};
$('corrFrameOther').onclick = () => {
  const target = S.corrTarget;
  if(!target) return;
  if(!target.pickingFrame){
    target.pickingFrame = true;      // scrub freely; the panel stays up
    renderCorrPanel();
    return;
  }
  target.pickingFrame = false;
  target.draft.frame_is_bounce = false;
  target.draft.frame = S.trackView.cursor;
  postDraft();
};

/* ---------- ball dot overlay: tap/drag to correct the position ---------- */
const CORR_DOT_GRAB_PX = 30;   // video-px hit radius for starting a drag

function corrDotFrame(){
  const target = S.corrTarget;
  if(!target || !target.draft.ball || target.draft.type === 'none') return null;
  return target.draft.frame_is_bounce === false && target.draft.frame != null
    ? target.draft.frame : target.frame;
}

function drawCorrectionDot(){
  const target = S.corrTarget;
  const frame = corrDotFrame();
  if(frame === null || S.trackView.cursor !== frame && !target.pickingFrame) return;
  const b = target.draft.ball;
  const r = Math.max(6, S.W / 160);
  ctx.beginPath();
  ctx.arc(b.x, b.y, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 200, 40, 0.35)';
  ctx.fill();
  ctx.lineWidth = Math.max(2, r / 4);
  ctx.strokeStyle = '#ffc828';
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(b.x, b.y, Math.max(2, r / 5), 0, Math.PI * 2);
  ctx.fillStyle = '#ffc828';
  ctx.fill();
}

let corrDotDrag = null;   // {pointerId} while dragging the dot

/* Kept alive in index.html ONLY while the dot did: nothing else called it. */
function canvasPointToVideo(e){
  const r = canvas.getBoundingClientRect();
  return {
    x: Math.min(S.W - 1, Math.max(0, (e.clientX - r.left) * S.W / r.width)),
    y: Math.min(S.H - 1, Math.max(0, (e.clientY - r.top) * S.H / r.height)),
  };
}

canvas.addEventListener('pointerdown', e => {
  if(S.phase !== 'track' || !S.corrTarget) return;
  const b = S.corrTarget.draft.ball;
  const frame = corrDotFrame();
  const p = canvasPointToVideo(e);
  const grabbable = b && frame !== null && S.trackView.cursor === frame;
  if(grabbable && Math.hypot(p.x - b.x, p.y - b.y) <= CORR_DOT_GRAB_PX){
    e.preventDefault();
    try{ canvas.setPointerCapture(e.pointerId); }catch(_){ /* synthetic events */ }
    corrDotDrag = { pointerId: e.pointerId };
    S.corrTarget.draft.ball = { x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 };
    redrawTrackCanvas();
  }
});
canvas.addEventListener('pointermove', e => {
  if(!corrDotDrag || e.pointerId !== corrDotDrag.pointerId || !S.corrTarget) return;
  e.preventDefault();
  const p = canvasPointToVideo(e);
  S.corrTarget.draft.ball = { x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 };
  redrawTrackCanvas();
});
function endCorrDotDrag(e){
  if(!corrDotDrag || e.pointerId !== corrDotDrag.pointerId) return;
  corrDotDrag = null;
  if(S.corrTarget) postDraft();
}
canvas.addEventListener('pointerup', endCorrDotDrag);
canvas.addEventListener('pointercancel', endCorrDotDrag);
