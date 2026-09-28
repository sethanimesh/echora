export const choices = {
  water: 'Please bring me water.', help: 'I need help, please.', yes: 'Yes.', no: 'No.',
};
export function hitTile(point, rects, inset = 18) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return Object.keys(rects).find(id => {
    const r=rects[id];
    return point.x >= r.left+inset && point.x <= r.right-inset && point.y >= r.top+inset && point.y <= r.bottom-inset;
  }) ?? null;
}
export function scoreTrial(hits, expected) {
  const total=hits.length, correct=hits.filter(id=>id===expected).length;
  return { total, correct, percent:total?Math.round(correct/total*100):0, passed:total>=12 && correct/total>=.7 };
}
export class TileDwell {
  target=null; since=0; fired=false;
  reset(){this.target=null;this.since=0;this.fired=false;}
  update(target,now,ms){
    if(target!==this.target){this.target=target;this.since=now;this.fired=false;}
    if(!target || this.fired)return {progress:0,selected:null};
    const progress=Math.min(1,(now-this.since)/ms);
    if(progress>=1){this.fired=true;return{progress:1,selected:target};}
    return{progress,selected:null};
  }
}

// WebGazer 3.5.3 retains 50 click samples. Nine targets × five frames = 45;
// pausing tracking must never reset the count and add duplicate target samples.
export class CalibrationSampler {
  count = 0;
  started = null;
  last = -Infinity;
  reset() { this.count = 0; this.pause(); }
  pause() { this.started = null; this.last = -Infinity; }
  take(now) {
    if (this.started === null) this.started = now;
    if (this.count >= 5 || now - this.started < 1000 || now - this.last < 450) return false;
    this.count++; this.last = now; return true;
  }
  get complete() { return this.count === 5; }
}
