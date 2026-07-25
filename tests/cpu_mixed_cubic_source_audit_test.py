from pathlib import Path


source = Path("src/clqr.cc").read_text()
begin = source.index("bool EliminateMixedStageWithMaps(")
end = source.index("\nbool StateBasisIsIdentity(", begin)
body = source[begin:end]

assert body.count("const Matrix r_times_y = old_R * basis.Y;") == 1
assert body.count("const Matrix r_times_z = old_R * basis.Z;") == 1
assert "r_times_y(u, col)" in body
assert body.count("r_times_z(u, col)") == 2
assert "for (std::size_t v = 0; v < m; ++v)" not in body
