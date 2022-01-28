# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def Create_Model():
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    mdb.Model(name='Model-5', modelType=STANDARD_EXPLICIT)

    s1 = mdb.models['Model-5'].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.rectangle(point1=(-12.0, -12.0), point2=(12.0, 12.0))
    p = mdb.models['Model-5'].Part(name='Part-1', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-5'].parts['Part-1']
    p.BaseSolidExtrude(sketch=s1, depth=0.75)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-5'].parts['Part-1']
    del mdb.models['Model-5'].sketches['__profile__']
    p1 = mdb.models['Model-5'].parts['Part-1']
    mdb.models['Model-5'].Material(name='Material-1')
    mdb.models['Model-5'].materials['Material-1'].Elastic(table=((30000000.0, 0.3), 
        ))
    mdb.models['Model-5'].HomogeneousSolidSection(name='Section-1', 
        material='Material-1', thickness=None)
    p = mdb.models['Model-5'].parts['Part-1']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    region = p.Set(cells=cells, name='Set-1')
    p = mdb.models['Model-5'].parts['Part-1']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    a = mdb.models['Model-5'].rootAssembly
    a1 = mdb.models['Model-5'].rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-5'].parts['Part-1']
    a1.Instance(name='Part-1-1', part=p, dependent=ON)
    p1 = mdb.models['Model-5'].parts['Part-1']
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#10 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(faces=pickedFaces, point1=p.InterestingPoint(
        edge=e1[0], rule=MIDDLE), point2=p.InterestingPoint(edge=e1[7], 
        rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#8 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point2=v1[1], faces=pickedFaces, 
        point1=p.InterestingPoint(edge=e[12], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#11 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(faces=pickedFaces, point1=p.InterestingPoint(
        edge=e1[12], rule=MIDDLE), point2=p.InterestingPoint(edge=e1[2], 
        rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#20 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(faces=pickedFaces, point1=p.InterestingPoint(
        edge=e[15], rule=MIDDLE), point2=p.InterestingPoint(edge=e[18], 
        rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    del p.features['Partition face-4']
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#20 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v2[0], faces=pickedFaces, 
        point2=p.InterestingPoint(edge=e1[15], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#41 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(faces=pickedFaces, point1=p.InterestingPoint(
        edge=e[21], rule=MIDDLE), point2=p.InterestingPoint(edge=e[2], 
        rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#840 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point2=v2[4], faces=pickedFaces, 
        point1=p.InterestingPoint(edge=e1[27], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#4000 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v1[6], faces=pickedFaces, 
        point2=p.InterestingPoint(edge=e[32], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#8001 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v2[3], faces=pickedFaces, 
        point2=p.InterestingPoint(edge=e1[30], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#1000 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v1[17], faces=pickedFaces, 
        point2=p.InterestingPoint(edge=e[36], rule=MIDDLE))
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#2001 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v2[3], point2=v2[5], faces=pickedFaces)
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#40000 ]', ), )
    v1, e, d1 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v1[22], point2=v1[5], faces=pickedFaces)
    p = mdb.models['Model-5'].parts['Part-1']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#80001 ]', ), )
    v2, e1, d2 = p.vertices, p.edges, p.datums
    p.PartitionFaceByShortestPath(point1=v2[3], point2=v2[16], faces=pickedFaces)
    a = mdb.models['Model-5'].rootAssembly
    a.regenerate()
    mdb.models['Model-5'].StaticStep(name='load', previous='Initial')
    a = mdb.models['Model-5'].rootAssembly
    e1 = a.instances['Part-1-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#8822 #584 ]', ), )
    region = a.Set(edges=edges1, name='Z-plane')
    mdb.models['Model-5'].DisplacementBC(name='BC-1', createStepName='Initial', 
        region=region, u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    a = mdb.models['Model-5'].rootAssembly
    e1 = a.instances['Part-1-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#92200011 #12 ]', ), )
    region = a.Set(edges=edges1, name='X-plane')
    mdb.models['Model-5'].DisplacementBC(name='BC-2', createStepName='Initial', 
        region=region, u1=SET, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    a = mdb.models['Model-5'].rootAssembly
    e1 = a.instances['Part-1-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#904400 #a00 ]', ), )
    region = a.Set(edges=edges1, name='Y-plane')
    mdb.models['Model-5'].DisplacementBC(name='BC-3', createStepName='Initial', 
        region=region, u1=UNSET, u2=SET, u3=UNSET, ur1=UNSET, ur2=UNSET, 
        ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)
    a = mdb.models['Model-5'].rootAssembly
    s1 = a.instances['Part-1-1'].faces
    side1Faces1 = s1.getSequenceFromMask(mask=('[#283807 ]', ), )
    region = a.Surface(side1Faces=side1Faces1, name='Surf-1')
    mdb.models['Model-5'].Pressure(name='Load-1', createStepName='load', 
        region=region, distributionType=UNIFORM, field='', magnitude=-400.0, 
        amplitude=UNSET)


def create_global():
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    mdb.Model(name='Model-5_global', objectToCopy=mdb.models['Model-5'])
    a = mdb.models['Model-5_global'].rootAssembly
    p1 = mdb.models['Model-5_global'].parts['Part-1']
    p = mdb.models['Model-5_global'].parts['Part-1']
    f1, e = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f1[10], sketchUpEdge=e[46], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 
        0.75))
    s1 = mdb.models['Model-5_global'].ConstrainedSketch(name='__profile__', 
        sheetSize=33.97, gridSpacing=0.84, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-5_global'].parts['Part-1']
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(-5.0, -1.0), point2=(5.0, 1.0))
    p = mdb.models['Model-5_global'].parts['Part-1']
    f, e1 = p.faces, p.edges
    p.CutExtrude(sketchPlane=f[10], sketchUpEdge=e1[46], sketchPlaneSide=SIDE1, 
        sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)
    s1.unsetPrimaryObject()
    del mdb.models['Model-5_global'].sketches['__profile__']
    p = mdb.models['Model-5_global'].parts['Part-1']
    p.seedPart(size=0.6, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, 
        kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    p = mdb.models['Model-5_global'].parts['Part-1']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
        elemType3))
    p = mdb.models['Model-5_global'].parts['Part-1']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=C3D20R)
    elemType2 = mesh.ElemType(elemCode=C3D15)
    elemType3 = mesh.ElemType(elemCode=C3D10)
    p = mdb.models['Model-5_global'].parts['Part-1']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
        elemType3))
    p = mdb.models['Model-5_global'].parts['Part-1']
    p.generateMesh()
    a = mdb.models['Model-5_global'].rootAssembly
    p1 = mdb.models['Model-5_global'].parts['Part-1']
    p = mdb.models['Model-5_global'].parts['Part-1']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f ]', ), )
    p.Set(faces=faces, name='cut_global')

    a.regenerate()
    mdb.Job(name='global_model', model='Model-5_global', description='', 
        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs['global_model'].writeInput(consistencyChecking=OFF)


def create_local():
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    a = mdb.models['Model-5'].rootAssembly
    mdb.Model(name='Model-5_local', objectToCopy=mdb.models['Model-5'])
    a = mdb.models['Model-5_local'].rootAssembly
    p1 = mdb.models['Model-5_local'].parts['Part-1']
    p = mdb.models['Model-5_local'].parts['Part-1']
    f1, e = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f1[10], sketchUpEdge=e[31], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 
        0.75))
    s = mdb.models['Model-5_local'].ConstrainedSketch(name='__profile__', 
        sheetSize=33.97, gridSpacing=0.84, transform=t)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-5_local'].parts['Part-1']
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    s.unsetPrimaryObject()
    del mdb.models['Model-5_local'].sketches['__profile__']
    p = mdb.models['Model-5_local'].parts['Part-1']
    f, e1 = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f[10], sketchUpEdge=e1[46], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 
        0.75))
    s1 = mdb.models['Model-5_local'].ConstrainedSketch(name='__profile__', 
        sheetSize=33.97, gridSpacing=0.84, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-5_local'].parts['Part-1']
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(-5.0, -1.0), point2=(5.0, 1.0))
    s1.unsetPrimaryObject()
    del mdb.models['Model-5_local'].sketches['__profile__']
    p = mdb.models['Model-5_local'].parts['Part-1']
    f1, e = p.faces, p.edges
    t = p.MakeSketchTransform(sketchPlane=f1[10], sketchUpEdge=e[46], 
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 
        0.75))
    s = mdb.models['Model-5_local'].ConstrainedSketch(name='__profile__', 
        sheetSize=33.97, gridSpacing=0.84, transform=t)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models['Model-5_local'].parts['Part-1']
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    s.rectangle(point1=(-12.0, -12.0), point2=(12.0, 12.0))
    s.rectangle(point1=(-5.0, -1.0), point2=(5.0, 1.0))
    p = mdb.models['Model-5_local'].parts['Part-1']
    f, e1 = p.faces, p.edges
    p.CutExtrude(sketchPlane=f[10], sketchUpEdge=e1[46], sketchPlaneSide=SIDE1, 
        sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
    s.unsetPrimaryObject()
    del mdb.models['Model-5_local'].sketches['__profile__']
    p = mdb.models['Model-5_local'].parts['Part-1']
    p.seedPart(size=0.5, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-5_local'].parts['Part-1']
    p.generateMesh()
    a = mdb.models['Model-5_local'].rootAssembly
    p1 = mdb.models['Model-5_local'].parts['Part-1']
    p = mdb.models['Model-5_local'].parts['Part-1']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f ]', ), )
    p.Set(faces=faces, name='cut_local')

    a.regenerate()
    mdb.Job(name='local_model', model='Model-5_local', description='', 
        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs['local_model'].writeInput(consistencyChecking=OFF)





