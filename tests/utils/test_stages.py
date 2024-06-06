from rail.utils.testing_utils import check_stage_params

import rail.stages
rail.stages.import_and_attach_all()
from rail.stages import *
import ceci



def test_all_stage_params():

    fail_list = []    
    
    for key, val in RailStage.pipeline_stages.items():
        fail_string = check_stage_params(val[0])
        if fail_string:
            fail_list.append(fail_string)
        
    if fail_list:
        for fail_ in fail_list:
            print(fail_)
        raise TypeError(f"Found {len(fail_list)} unparseable parameters")
        
