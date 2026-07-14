#!/usr/bin/env bash

# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compatibility entrypoint for existing repository checkouts. The complete
# installer is intentionally kept in the repository root so public installation
# never needs to download and execute a second script.

set -o errexit
set -o nounset
set -o pipefail

script_source=${BASH_SOURCE[0]:-}
if [ -z "$script_source" ] || [ "$script_source" = "-" ] || [ ! -f "$script_source" ]; then
  echo "scripts/installation/install.sh is a checkout-only compatibility entrypoint." >&2
  echo "Use the repository-root install.sh for downloads and piped installation." >&2
  exit 2
fi

script_dir=$(cd "$(dirname "$script_source")" && pwd)
root_installer="$script_dir/../../install.sh"
if [ ! -f "$root_installer" ]; then
  echo "repository-root install.sh not found: $root_installer" >&2
  exit 2
fi

exec bash "$root_installer" "$@"
