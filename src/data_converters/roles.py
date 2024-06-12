"""
File: src/data_converters/roles.py
Creation Date: 2024-06-07

Contains role class definitions for various role sets. With respect to our research,
we only trained and evaluated models using the 2args3mods setting, without frames and animations.
However, this file was created to mimic the original codebase, which contained definitions for all
roles. Future work can extend this file.

THIS ONLY SUPPORTS V2 AT THE MOMENT!
"""


class Roles:
    has_role_other = True
    has_mod_other = False

    # v2 uses PRD, while v1 uses V
    ROLE_PRD = "PRD"

    ROLE_OTHER = "ARG-OTHER" if has_role_other else None  # make it None if not using ROLE_OTHER

    ROLE_SET = {
        'A0': 5,
        'A1': 1,
        'A2': 4,
        'A3': 10,
        'A4': 14,
        'A5': 28,
        'AM': 25,
        'AM-ADV': 12,
        'AM-CAU': 8,
        'AM-DIR': 18,
        'AM-DIS': 6,
        'AM-EXT': 24,
        'AM-LOC': 2,
        'AM-MNR': 7,
        'AM-MOD': 21,
        'AM-NEG': 13,
        'AM-PNC': 15,
        'AM-PRD': 27,
        'AM-TMP': 3,
        'C-A1': 26,
        'C-V': 9,
        'R-A0': 11,
        'R-A1': 17,
        'R-A2': 20,
        'R-A3': 29,
        'R-AM-CAU': 23,
        'R-AM-LOC': 16,
        'R-AM-MNR': 19,
        'R-AM-TMP': 22,
        '<OTHER>': 30
    }

    @staticmethod
    def is_modifier(role):
        return (((len(role) >= 2) and (role[:2] == "AM"))
                or ((len(role) >= 4) and (role[:4] == "ARGM")))

    @staticmethod
    def adjust_role(role):
        """
        This method is used to adjust the role if necessary e.g. for frames and animations.
        :param role: The role tag to adjust
        :return:
        """
        return role


class Roles2Args3Mods(Roles):
    has_role_other = False
    has_mod_other = False
    ROLE_SET = {'ARG0': 0, 'ARG1': 1, 'ARGM-LOC': 2,
                'ARGM-TMP': 3, 'ARGM-MNR': 4, Roles.ROLE_PRD: 5}
