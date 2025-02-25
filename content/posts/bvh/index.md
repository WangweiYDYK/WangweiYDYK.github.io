BVTT front type

bvh的三种操作：build，refit和update

GPU中不依赖父子节点关系就可以找到父子节点，通过Morton Code

external nodes
internal nodes
```cpp
	void LBvhFixedDeformable::build() {
		/// calculate scene bounding box
		BOX	bv{};
		checkCudaErrors(cudaMemcpy(cbvh().bv(), &bv, sizeof(BOX), cudaMemcpyHostToDevice));
#if MACRO_VERSION
			configuredLaunch({ "CalcBVARCSim", cbvh().primSize() }, calcMaxBVARCSim,
				cbvh().primSize(), d_bxsARCSim, cbvh().bv());
#else
			configuredLaunch({ "CalcBV", cbvh().primSize() }, calcMaxBV,
				cbvh().primSize(), (const int3*)d_faces, (const PointType*)d_vertices, cbvh().bv());
#endif
		checkCudaErrors(cudaMemcpy(&bv, cbvh().bv(), sizeof(BOX), cudaMemcpyDeviceToHost));

		//Logger::tick<TimerType::CPU>();
		//_mortonCoder.configureScene(bv._min, bv._max);
		//Logger::tock<TimerType::CPU>("ConfigureCoder");
		//configuredLaunch({ "CalcEMCs", cbvh().primSize() }, calcEMCs,
		//	cbvh().primSize(), d_faces, d_vertices, _mortonCoder.getCoder(), getRawPtr(d_keys64));
#if MACRO_VERSION
			configuredLaunch({ "CalcMCsARCSim", cbvh().primSize() }, calcMCsARCSim,
				cbvh().primSize(), d_bxsARCSim, bv, getRawPtr(d_keys32));
#else
			configuredLaunch({ "CalcMCs", cbvh().primSize() }, calcMCs,
				cbvh().primSize(), d_faces, d_vertices, bv, getRawPtr(d_keys32));
#endif
		//configuredLaunch({ "CalcMC64s", cbvh().primSize() }, calcMC64s,
		//	cbvh().primSize(), d_faces, d_vertices, cbvh().bv(), getRawPtr(d_keys64));

		reorderPrims();

		/// build primitives
#if MACRO_VERSION
			configuredLaunch({ "BuildPrimsARCSim", cbvh().primSize() }, buildPrimitivesARCSim,
				cbvh().primSize(), cbvh().lvs().getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_facesARCSim, d_bxsARCSim);
#else
			configuredLaunch({ "BuildPrims", cbvh().primSize() }, buildPrimitives,
				cbvh().primSize(), cbvh().lvs().getPrimitiveArray().portobj<0>(), getRawPtr(d_primMap), d_faces, d_vertices);
#endif

		/// build external nodes
		cbvh().intSize() = (cbvh().extSize() = cbvh().lvs().buildExtNodes(cbvh().primSize())) - 1;
		cbvh().lvs().calcSplitMetrics(cbvh().extSize());
		/// build internal nodes
		_unsortedTks.clearIntNodes(cbvh().intSize());
		configuredLaunch({ "BuildIntNodes", cbvh().extSize() }, buildIntNodes,
			cbvh().extSize(), getRawPtr(d_count), cbvh().lvs().portobj<0>(), _unsortedTks.portobj<0>());

Logger::recordSection<TimerType::GPU>("construct_bvh");

		/// first correct indices, then sort
		reorderIntNodes();

Logger::recordSection<TimerType::GPU>("sort_bvh");

		//_unsortedTks.scatter(cbvh().intSize(), getRawPtr(d_tkMap), cbvh().tks());

		printf("Primsize: %d Extsize: %d\n", cbvh().primSize(), cbvh().extSize());
	}
```