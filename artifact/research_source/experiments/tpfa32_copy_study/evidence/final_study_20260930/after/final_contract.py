"""Non-numerical fixed final contract; does not alter the historical draft."""
import copy
from core import validate_config, ANALYSIS_RULE

SELECTION_SHA='3ebc9cdc4b7e94e78b09460939db7959f93979dffe1dbf7f0320c50449e55d75'


def validate_final_config(config):
    historical=copy.deepcopy(config);historical['analysis_rule']=ANALYSIS_RULE
    validate_config(historical)
    if config['status']!='locked' or config['analysis_rule']!='t10_adv975_eq95':raise ValueError('final lock missing')
    if config.get('fixture_only') or config.get('review_pending'):raise ValueError('fixture/provisional configuration refused')
    if config.get('selection')!=dict(primary='R-a',development_sha256=SELECTION_SHA,
        recipes={'R-a':dict(updates=15360,base_updates=15360,cooldown_updates=1536,endpoint_updates=16896)}):
        raise ValueError('fixed selected replacement changed')
    if any(config[k]!=128 for k in ('development_controls','calibration_controls','final_controls')):
        raise ValueError('fixed common evaluation population changed')
    return config
