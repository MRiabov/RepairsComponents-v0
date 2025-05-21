from bd_warehouse.pipe import Pipe, Nps, Material, Identifier
from build123d import Edge, Wire, export_gltf


def create_pipe(
    nps: str | Nps,
    material: str | Material,
    identifier: str | Identifier,
    pipe_path: Edge | Wire,
    file_path: str | None = None,
):
    """
    Create a pipe with the given parameters.

    Parameters
    ----------
    nps : str | Nps
        Nominal Pipe Size (NPS). See bd_warehouse.pipe.Nps for possible values.
    material : str | Material
        Material of the pipe. See bd_warehouse.pipe.Material for possible values.
    identifier : str | Identifier
        A unique identifier for the pipe. See bd_warehouse.pipe.Identifier for possible values.
    path : Edge | Wire
        Path of the pipe - created from Sketch or Wire of build123d.

    Returns
    -------
    pipe : Pipe
        The resulting pipe
    """
    pipe = Pipe(nps=nps, material=material, identifier=identifier, path=pipe_path)  # type: ignore
    if file_path:
        export_gltf(pipe, file_path)
    return pipe
