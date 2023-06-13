"""
 Summary
 -------
 functions for accessing hyp3.
 
 Notes
 -----
 Created By:  Jason Herning
"""

from getpass import getpass
from typing import Tuple

from hyp3_sdk import HyP3, exceptions


def get_netrc_credentials() -> Tuple[str, str]:
    """
    Get earthdata credentials from a .netrc file in the projects root dir.

    Returns
    --------
    username : str
        The user's earthdata username.
    password : str
        The user's earthdata password.
    """

    with open(".netrc", "r") as f:
        contents = f.read()
    username = contents.split(" ")[3]
    password = contents.split(" ")[5].split("\n")[0]

    return username, password


def input_earthdata_login() -> Tuple[str, str]:
    """
    Get a user's earthdata credentials from command-line input.

    Returns
    --------
    username : str
        The user's earthdata username.
    password : str
        The user's earthdata password.
    """

    print("Enter your NASA EarthData username: ", end="")
    username = input()
    password = getpass()
    return username, password


def write_netrc(username, password) -> None:
    """
    Write the earthdata username and password to a .netrc file.

    Parameters
    -----------
    username : str
        The user's earthdata username.
    password : str
        The user's earthdata password.

    Returns:
    --------
    None
    """

    with open(".netrc", "w+") as f:
        f.write(
            "machine urs.earthdata.nasa.gov login"
            + username
            + "password"
            + password
            + "\n"
        )


def hyp3_login() -> HyP3:
    """
    Pull the user's earthdata credentials from the .netrc file, else prompt them for it,
    and start a HyP3 session.

    Returns
    --------
    hyp3 : HyP3
        The hyp3 session class.
    """

    try:
        username, password = get_netrc_credentials()
    except IndexError and FileNotFoundError:
        pass

    try:
        username, password = input_earthdata_login()
        hyp3 = HyP3(username=username, password=password)
    except exceptions.AuthenticationError:
        print("try again!")
        hyp3 = hyp3_login()

    write_netrc(username, password)
    return hyp3
