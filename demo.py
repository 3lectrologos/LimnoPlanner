#! /usr/bin/env python

import util
import plan


fn = 'var/tc_limnolog-00110714-135138_bgape_gp.mat'
tc = util.Testcase.from_mat(fn)
p = plan.Planner(tc)
p.run(4, record=True)
