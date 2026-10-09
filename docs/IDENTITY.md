# Identity and contribution policy

All contributions to **DD** (<https://github.com/SamerKharboush/DD>) are authored
and signed under **Samer Kharboush**
(<https://github.com/SamerKharboush>).

This policy covers contributions made *to this fork*. It does not alter CellType
Inc.'s rights in code inherited from upstream — see `NOTICE`.

## Enforced git identity

These values are set globally and enforced by a `pre-commit` hook in this
repository. A commit that does not match is **rejected**:

    user.name       Samer Kharboush
    user.email      88907921+SamerKharboush@users.noreply.github.com
    commit.gpgsign  true
    tag.gpgsign     true
    user.signingkey A7352383D82981985CA0431657800357C717D972

The **noreply** address is required — it is what makes GitHub link commits to the
account rather than to an unverified mailbox.

## GPG signing

All commits and tags are GPG-signed with an RSA 4096 key:

    fingerprint  A7352383D82981985CA0431657800357C717D972
    uid          Samer Kharboush <88907921+SamerKharboush@users.noreply.github.com>

Public key: `samer_pubkey.asc` (repository root). Register it at
<https://github.com/settings/signing-keys> to obtain GitHub's green **Verified**
badge. Until it is registered, commits still carry a valid cryptographic
signature — GitHub simply will not display the badge.

Verify any commit in this repository with:

    git log --show-signature -1

A good signature reads `Good signature from "Samer Kharboush"`.

## Copyright headers

New source files carry:

    # Copyright (c) 2026 Samer Kharboush
    # SPDX-License-Identifier: MIT

Files derived from upstream carry **both** notices. MIT requires that the
upstream copyright notice travel with substantial portions of the software, so
rebranding must never erase CellType Inc.'s line:

    # Copyright (c) 2026 CellType Inc.
    # Copyright (c) 2026 Samer Kharboush
    # SPDX-License-Identifier: MIT

## Why dual notices

Upstream `LICENSE` is MIT, Copyright (c) 2026 CellType Inc. This fork is
redistributed under the same terms. Removing or obscuring the upstream notice
would place the fork outside the terms under which it was received. Provenance
is not optional; it is a licence condition.

## Local setup

    git config --local user.name  "Samer Kharboush"
    git config --local user.email "88907921+SamerKharboush@users.noreply.github.com"
    git config --local commit.gpgsign true
    git config --local tag.gpgsign    true
    git config --local user.signingkey A7352383D82981985CA0431657800357C717D972