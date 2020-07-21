import unittest
import numpy
import os
import subprocess
import tempfile
import shutil
import zipfile

from qcip_tools import derivatives


def array_almost_equals(a, b, places=7, delta=None, msg=''):
    """Check if two arrays containing float number are almost equals"""

    atol = 10 ** (-places)

    if delta is not None:
        atol = delta

    return numpy.testing.assert_allclose(a, b, atol=atol, err_msg=msg)


class NachosTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tests_files_directory = os.path.join(os.path.dirname(__file__), 'tests_files')
        self.temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)

    def assertArraysAlmostEqual(self, a, b, places=3, delta=None, msg=''):
        return array_almost_equals(a, b, places=places, delta=delta, msg=msg)

    def assertTensorsAlmostEqual(self, a, b, places=3, delta=None, msg='', skip_frequency_test=False):
        """Check if two tensors are the same
        """

        self.assertEqual(
            a.representation.representation(),
            b.representation.representation(),
            msg='representations does not matches ({} != {})'.format(
                a.representation.representation(), b.representation.representation()))

        if derivatives.is_geometrical(a.representation):
            self.assertEqual(
                a.spacial_dof,
                b.spacial_dof,
                msg='spacial dof does not matches ({} != {})'.format(a.spacial_dof, b.spacial_dof))

        if derivatives.is_electrical(a.representation) and not skip_frequency_test:
            self.assertEqual(
                a.frequency,
                b.frequency,
                msg='frequencies does not matches ({} != {})'.format(a.frequency, b.frequency))

        return self.assertArraysAlmostEqual(a.components, b.components, places, delta, msg)

    def copy_to_temporary_directory(self, path, new_name='', directory=None):
        """Copy the content of a file to the temporary directory

        :param path: path to the file to copy
        :type path: str
        :param new_name: the new name of the file in the temporary directory (if blank, the one from file_ is used)
        :type new_name: str:
        :param directory: directory in which the file should be copied. If nothing is provided, it is the temporary dir
        :type directory: str
        :rtype: str
        """

        path_in_test = os.path.join(self.tests_files_directory, path)

        if not os.path.exists(path_in_test):
            raise FileNotFoundError(path_in_test)

        if not new_name:
            new_name = os.path.basename(path)

        if not directory:
            directory = self.temporary_directory

        path_in_temp = os.path.join(directory, new_name)

        if os.path.exists(path_in_temp):
            raise FileExistsError(path_in_temp)

        with open(path_in_temp, 'wb') as f:
            with open(os.path.join(self.tests_files_directory, path), 'rb') as fx:
                f.write(fx.read())

        return path_in_temp

    def setup_temporary_directory(self, directory=None):
        """Create a temporary directory

        :param directory: temporary in which the subdirectory is created (if nothing, temporary directory is used)
        :type directory: str
        :rtype: str
        """

        if not directory:
            directory = self.temporary_directory

        return tempfile.mkdtemp(dir=directory)

    def run_python_script(
            self,
            path,
            args=None,
            cwd='.',
            in_pipe=subprocess.DEVNULL,
            out_pipe=subprocess.DEVNULL,
            err_pipe=subprocess.DEVNULL):
        """
        Return a subprocess.Popen object to a python process with the given script executed

        :param path: path to the script, with respect to the top of the package
        :type path: str
        :param args: the args
        :type args: list
        :param cwd: current directory
        :type cwd: str
        :param in_pipe: the input pipe (set to subprocess.PIPE for communication)
        :param out_pipe: the output pipe (set to subprocess.PIPE for communication)
        :param err_pipe: the error pipe (set to subprocess.PIPE for communication)
        :rtype: subprocess.Popen
        """

        real_path = os.path.join(cwd, path)
        if not os.path.isfile(real_path):
            raise FileNotFoundError(real_path)

        cmd = ['python', path]
        if args:
            cmd.extend(args)

        return subprocess.Popen(cmd, stdin=in_pipe, stdout=out_pipe, stderr=err_pipe, cwd=cwd)

    def unzip_it(self, path, directory=None):
        zf = zipfile.ZipFile(path, 'r')

        if not directory:
            directory = self.temporary_directory

        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

        zf.extractall(directory)
        zf.close()
