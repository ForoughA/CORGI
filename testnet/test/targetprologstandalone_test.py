import sys
import unittest

from targetprologstandalone import entry_point

from StringIO import StringIO


class TestSpyrolog(unittest.TestCase):

    def test_direct(self):
        stdout_bak = sys.stdout

        try:
            result = StringIO()
            sys.stdout = result
            entry_point([
                '',
                'data/direct_facts.txt',
                'data/direct_sims.txt',
                #'s0(e0,e1).|s0(e2,e1).|s0(e3,e1).|s0(e4,e1).|s0(e5,e1).|s0(e6,e1).|s0(e1,e0).|s0(e1,e2).|s0(e1,e3).|s0(e1,e4).|s0(e1,e5).|s0(e1,e6).',
                's0(e1,e3).',
                '3',
                '0',
                'prod|prod',
                '0'
            ])
            
            result = result.getvalue().strip()
            print "result", result
            self.assertIn('0.5', result)
        finally:
            sys.stdout = stdout_bak



if __name__ == '__main__':
    unittest.main()
