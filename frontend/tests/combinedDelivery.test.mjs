import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { tones, speechPaces } from '../components/echora/delivery.ts';
import { combinedDelivery } from '../components/echora/deliveryMerge.ts';
const require=createRequire(import.meta.url), exports={};
const code=ts.transpileModule(readFileSync(new URL('../components/echora/CombinedDelivery.tsx',import.meta.url),'utf8'),{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.CommonJS}}).outputText;
new Function('require','exports',code)(id=>id==='./FaceComparison'?{default:()=>null}:id==='./delivery'?{tones,speechPaces}:id==='./deliveryMerge'?{combinedDelivery}:require(id),exports);
const voice={state:'ready',tone:'firm',rate:1.2};
const face={state:'ready',visibility:'clear_face',cue:'smile',tone:'warm'};
function render(options={}) {
  return renderToStaticMarkup(React.createElement(exports.default,{voice,face,currentRate:1,disabled:false,onAccept(){},...options}));
}
test('Echora resolves disagreement into one final tone and pace with no source selector',()=>{
 let applications=0;
 const html=render({onAccept:()=>applications++});
 assert.equal(applications,0);
 assert.equal((html.match(/Use suggested delivery/g)||[]).length,1);
 assert.match(html,/Warm/);assert.match(html,/Standard \(1×\)/);
 assert.doesNotMatch(html,/The cues differ|Which tone|aria-pressed|<fieldset|Voice tone:|Facial cue:|Faster/);
});
test('no partial final suggestion or apply action while the other cue is pending',()=>{
 const html=render({face:{state:'analyzing'}});
 assert.match(html,/Combining the cues/);
 assert.doesNotMatch(html,/Suggested delivery:|Use suggested delivery/);
});
test('a failed face check keeps a usable voice result and one apply action',()=>{
 const html=render({face:{state:'error',message:'Camera unavailable'}});
 assert.match(html,/Camera unavailable/);assert.match(html,/Firm/);assert.match(html,/Faster/);
 assert.equal((html.match(/Use suggested delivery/g)||[]).length,1);
});
test('unclear results preserve settings without offering an empty suggestion',()=>{
 const html=render({voice:{state:'ready',tone:null,rate:null},face:{...face,cue:'unclear'}});
 assert.match(html,/No clear delivery suggestion/);assert.doesNotMatch(html,/Use suggested delivery/);
});
