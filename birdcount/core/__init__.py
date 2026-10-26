"""Code shared by `birdcount.deploy` and `birdcount.training`.

Nothing is re-exported here: import the submodule you need
(`birdcount.core.settings`, ...). A re-export would make every importer of
`birdcount.core` pay for the heaviest submodule's imports.
"""
