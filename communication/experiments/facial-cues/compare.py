"""Run a reproducible three-image trial; explicit compare/gemini modes upload images."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots',nargs=3,type=Path)
    parser.add_argument('--mode',choices=['local','gemini','compare'],default='local')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    os.environ['ECHORA_FACE_MODE']=args.mode
    from communication.backend import app, face_analysis, local_faces
    from communication.backend.providers import ProviderFailure
    frames=[]
    for path in args.snapshots:
        if path.stat().st_size>256*1024: parser.error('Each JPEG must be at most 256 KiB.')
        data=path.read_bytes()
        if not data.startswith(b'\xff\xd8\xff') or not data.endswith(b'\xff\xd9'): parser.error('Use JPEG snapshots.')
        frames.append(data)
    async def run():
        await face_analysis.prepare()
        return await face_analysis.suggest(frames)
    try: result=asyncio.run(run())
    except ProviderFailure as exc: parser.exit(1,exc.message+'\n')
    report=json.dumps({'mode':args.mode,'result':result,'note':'Model scores and timing only; not measured emotion accuracy.'},indent=2)+'\n'
    if args.output: args.output.write_text(report)
    else: print(report,end='')

if __name__=='__main__': main()
