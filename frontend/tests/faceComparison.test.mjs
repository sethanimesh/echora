import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { faceReport } from '../components/echora/faceReport.ts';
const require=createRequire(import.meta.url), exports={};
const code=ts.transpileModule(readFileSync(new URL('../components/echora/FaceComparison.tsx',import.meta.url),'utf8'),{compilerOptions:{jsx:ts.JsxEmit.ReactJSX,module:ts.ModuleKind.CommonJS}}).outputText;
new Function('require','exports',code)(id=>id==='./faceReport'?{faceReport}:require(id),exports);
const comparison={local:{state:'ready',cue:'positive_expression',visibility:'clear_face',elapsed_ms:30},gemini:{state:'ready',cue:'smile',visibility:'clear_face',elapsed_ms:2000},agreement:'same_style',selected:'gemini',policy:'paired-face-1'};
test('paired report shows both timings and fixed baseline without source selection',()=>{
 const html=renderToStaticMarkup(React.createElement(exports.default,{comparison}));
 assert.match(html,/0.03 s/);assert.match(html,/2.00 s/);assert.match(html,/Gemini remains/);
 assert.match(html,/Save comparison report/);assert.doesNotMatch(html,/Use camera|Use local|Use Gemini/);
});
test('report exporter whitelists outputs and excludes media, words and identity',()=>{
 const result=faceReport({...comparison,transcript:'private-word',local:{...comparison.local,image:'private-pixels',audio:'private-audio',identity:'private-person',frames:[{cue:'unclear',visibility:'clear_face',label:'Neutral',score:.4,margin:.1,image:'private-frame'}]}},'neutral','fits');
 const json=JSON.stringify(result);
 assert.doesNotMatch(json,/private-/);assert.equal(result.trial,'neutral');
 assert.equal(result.comparison.local.frames[0].score,.4);
});
test('local failure is visible while cloud reference remains available',()=>{
 const html=renderToStaticMarkup(React.createElement(exports.default,{comparison:{...comparison,local:{state:'error',message:'Local model unavailable'},agreement:'not_comparable'}}));
 assert.match(html,/Local model unavailable/);assert.match(html,/Gemini Flash/);assert.match(html,/not two clear styles/);
});
