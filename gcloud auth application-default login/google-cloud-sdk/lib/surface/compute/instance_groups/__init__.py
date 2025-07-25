# -*- coding: utf-8 -*- #
# Copyright 2014 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Commands for reading and manipulating instance groups."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.calliope import base


@base.ReleaseTracks(
    base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class InstanceGroups(base.Group):
  """Read and manipulate Compute Engine instance groups.

  Read and manipulate Compute Engine instance groups. To accommodate the
  differences between managed and unmanaged instances, some commands (such as
  `delete`) are in the managed or unmanaged subgroups.
  """


InstanceGroups.category = base.INSTANCES_CATEGORY

InstanceGroups.detailed_help = {
    'DESCRIPTION': """
        Read and manipulate Compute Engine instance groups.

        To accommodate the differences between managed and unmanaged instances,
        some commands (such as `delete`) are in the managed or unmanaged
        subgroups.

        For more information about instance groups, see the
        [instance groups documentation](https://cloud.google.com/compute/docs/instance-groups/).

        See also: [Instance groups API](https://cloud.google.com/compute/docs/reference/rest/v1/instanceGroups).
    """,
}
