# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import re
from copy import copy


_smiles_vocab = """H
He
Li
Be
B
C
N
O
F
Ne
Na
Mg
Al
Si
P
S
Cl
Ar
K
Ca
Sc
Ti
V
Cr
Mn
Fe
Co
Ni
Cu
Zn
Ga
Ge
As
Se
Br
Kr
Rb
Sr
Y
Zr
Nb
Mo
Tc
Ru
Rh
Pd
Ag
Cd
In
Sn
Sb
Te
I
Xe
Cs
Ba
La
Ce
Pr
Nd
Pm
Sm
Eu
Gd
Tb
Dy
Ho
Er
Tm
Yb
Lu
Hf
Ta
W
Re
Os
Ir
Pt
Au
Hg
Tl
Pb
Bi
Po
At
Rn
Fr
Ra
Ac
Th
Pa
U
Np
Pu
Am
Cm
Bk
Cf
Es
Fm
Md
No
Lr
Rf
Db
Sg
Bh
Hs
Mt
Ds
Rg
Cn
Nh
Fl
Mc
Lv
Ts
Og
b
c
n
o
s
p
0
1
2
3
4
5
6
7
8
9
[
]
(
)
.
=
#
-
+
\\
/
:
~
@
?
>
*
$
%""".splitlines()


NUM_REACTIONS = 115


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    MOL_START = 2
    MOL_END = 3
    # MOL: 4~156 / RXN: 157~271
    RXN_MIN = MOL_END + len(_smiles_vocab) + 1  # 157
    RXN_MAX = RXN_MIN + NUM_REACTIONS - 1       # 271
    

_smiles_token_to_id = {token: i for i, token in enumerate(_smiles_vocab, start=1)}
_smiles_token_max = max(_smiles_token_to_id.values())
_smiles_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted(_smiles_vocab, reverse=True))) + ")")

_smiles_id_to_token = {i: token for i, token in enumerate(_smiles_vocab, start=TokenType.MOL_END + 1)}

# for finetuning
_token_id_to_token = copy(_smiles_id_to_token)
_token_id_to_token[TokenType.END] = 'END'
# separate BBs and RXNs with ','
_token_id_to_token[TokenType.MOL_START] = _token_id_to_token[TokenType.MOL_END] = ','
for idx in range(TokenType.RXN_MIN, TokenType.RXN_MAX + 1):
    # ',' for rare cases of missing MOL_START
    _token_id_to_token[idx] = f'R{idx - TokenType.RXN_MIN},'


def tokenize_smiles(s_in: str):
    tok: list[int] = []
    for token in _smiles_token_pattern.findall(s_in):
        tok.append(_smiles_token_to_id.get(token, _smiles_token_max + 1))
    return tok


def decode_smiles(smiles_ids):
    smiles = [_smiles_id_to_token.get(int(smiles_id)) for smiles_id in smiles_ids]
    return ''.join([s for s in smiles if s is not None])


# for finetuning
def decode_tokens(token_ids):
    tokens = [_token_id_to_token.get(int(token_id)) for token_id in token_ids]
    sequence = ''.join([t for t in tokens if t is not None])
    sequence = sequence.split('END')[0]
    sequence = sequence.strip(',')  # ignore first and last commas
    sequence = re.sub(r',+', ',', sequence) # ,, -> ,
    return sequence
