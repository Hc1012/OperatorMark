"""TPFA32 copied-versus-from-scratch runner. Default: non-numerical plan."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from core import HERE, read, validate_config, authorize_campaign, source_hashes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='plan',choices=['plan','preflight','development','final'])
    parser.add_argument('--config',type=Path,default=HERE/'draft.json')
    parser.add_argument('--authority',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    config=validate_config(read(args.config))
    if args.command in ('plan','preflight'):
        print(json.dumps(dict(status=config['status'],owners=config['owners'],
                              oracle_budget=config['oracle_budget'],source_sha256=source_hashes(),
                              numerical_work=False,final_allowed=False,
                              review_pending=config['review_pending']),indent=2))
        return 0
    authority=read(args.authority) if args.authority else None
    authorize_campaign(config,authority,args.command)
    if args.output is None: raise ValueError('fresh private output directory required')
    from campaign import Campaign
    Campaign(config,authority,args.output,args.command).execute()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
