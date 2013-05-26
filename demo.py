#! /usr/bin/env python

import util
import plan


tc = util.test2()
p = plan.Planner(tc)
p.run(3, record=True)
