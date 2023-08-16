
'''
A class to print status messages using the least amount of space on a line. The
message header is printed as often as desired. Status values may be omitted.

Author:
    P. David, U.S. Army Research Laboratory
'''

import re
from phutils import font


class StatusMsg:

    def __init__(self, header, fmt, hdrint=5, spaces=1):
        '''
        Initialize the StatusMsg object.

        Arguments:
            header: (list) List of strings, each is the heading for one column
            in the output.

            fmt: (list) List of strings, each gives the print format for one
            column in the output. Any valid string.format strings are allowed,
            excluding the {:} characters. Example: ['<3d', '2s', '^5.2f'].

            hdrint: (int) How often should the header line be printed?
            Default = 5.

            space: (int) Number of spaces between columns. Default = 1.
        '''

        self.header = header     # column headings
        self.numvals = len(fmt)  # number of values to print
        self.hdrint = hdrint     # header print interval
        self.hdrskip = hdrint    # number of lines printed since last header
        self.fmt = []            # list of formats, one for each value
        self.fmt_none = []       # format to use when no value present (print '')

        for k, f in enumerate(fmt):
            # print('Fmt # {} = "{}"'.format(k, f))
            m = re.search('[0123456789]+', f)
            l = int(m[0]) if m else 6        # field length
            m = re.search('[<>^]', f)
            a = m[0] if m else '^'           # field alignment (default: center)

            self.fmt.append('{:' + fmt[k] + '}' + ' '*spaces)
            self.fmt_none.append('{:' + a + str(l) + 's}' + ' '*spaces)


    def reset(self):
        '''
        Reset the line counter so the header is printed on the next call to
        StatusMsg.print()
        '''
        self.hdrskip = self.hdrint


    def print(self, values):
        '''
        Print one line of output.

        Arguments:
            values: (list) A list of values to print. The length of `values`
            must always be identical to the number of formats given in the
            initialization. However, any number of values may be None, in which
            case these values are skipped over (spaces are printed).
        '''

        if self.hdrskip == self.hdrint:
            # Print the column headers.
            self.hdrskip = 1
            for k in range(self.numvals):
                print(self.fmt_none[k].format(self.header[k]), end='')
                # print(self.fmt_none[k].format(font(self.header[k],fbt='r;;b')), end='')
            print('')
        else:
            self.hdrskip += 1

        # Print the data values.
        for k in range(self.numvals):
            if values[k] is None:
                print(self.fmt_none[k].format(''), end='')
            else:
                print(self.fmt[k].format(values[k]), end='')
        print('')


    def message(self, values):
        '''
        Get the current status message as a string.

        Arguments:
            values: (list) A list of values to print. The length of `values`
            must always be identical to the number of formats given in the
            initialization. However, any number of values may be None, in which
            case these values are skipped over (spaces are printed).
        '''
        msg = ''
        for k in range(self.numvals):
            if values[k] is None:
                msg += self.fmt_none[k].format('')
            else:
                msg += self.fmt[k].format(values[k])
        return msg


    def csv(self, values, sep=','):
        '''
        Get the current status message as a CSV (comman-separated value) string.

        Arguments:
            values: (list) A list of values to print. The length of `values`
            must always be identical to the number of formats given in the
            initialization. However, any number of values may be None, in which
            case these values are skipped over (spaces are printed).

            sep: (str) The value separator.
        '''
        msg = ''
        for k in range(self.numvals):
            if values[k] is None:
                s = self.fmt_none[k].format('')
            else:
                s = self.fmt[k].format(values[k])
            if k > 0:
                msg += sep
            msg += s.strip(' ')
        return msg


if __name__ == '__main__':
    import numpy as np
    n = 7
    sm = StatusMsg(['Frame']+['TEST']*n, ['^10d']+['^10.2f']*n, hdrint=5, spaces=2)
    for k in range(20):
        r = 100*np.random.rand(n)
        values = [k] + list(r)              #  list of frame # and random values
        values[1 + np.random.randint(n)] = None             #  one value is None
        sm.print(values)
