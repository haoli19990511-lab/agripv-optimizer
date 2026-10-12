"""Offline logic tests. No claim of PCSE/Radiance field or physics validation."""
import ast
from dataclasses import dataclass, make_dataclass
import json
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import bridge
from sampling_core import SamplingConfig, make_sampling_plan, design_grid, compare_sampling
from weighted_engine import WeightedEngine, EngineConfig, economic_metrics, load_price_profile

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Design:
    tilt: float = 25.
    pitch: float = 7.
    azimuth: float = 180.
    clearance_height: float = 3.
    xcellgap: float = .0004
    ycellgap: float = .00235
    xgap_conventional: float = .20

MODULE = SimpleNamespace(module_x=1.134,module_y=2.278,numcellsx=6,numcellsy=24,xcell=.182,ycell=.091)
ARRAY = SimpleNamespace(n_mods=5,n_rows=5)

def example_weather(days=60):
    hours = pd.date_range("2001-01-01", periods=days*24, freq="h", tz="Etc/GMT-1")
    ghi = np.maximum(0, np.sin((hours.hour.to_numpy()-6)*np.pi/12))*500
    ghi *= 1+.3*np.sin(np.arange(len(hours))//24)
    return hours, ghi, ghi*.3

class SamplingTests(unittest.TestCase):
    def test_horizontal_energy_conserved_all_modes(self):
        h,g,d = example_weather()
        source = pd.Series(g,index=h)
        for mode in ("full","representative_hours","representative_days","combined"):
            p = make_sampling_plan(h,g,d,SamplingConfig(mode))
            actual = p.sample_hour.map(source).fillna(0).to_numpy()*p.scale.to_numpy()
            np.testing.assert_allclose(actual,g,atol=1e-10)
            self.assertEqual(len(p),len(h))
            self.assertTrue((p.scale >= 0).all())

    def test_full_is_identity(self):
        h,g,d=example_weather(2)
        p=make_sampling_plan(h,g,d,SamplingConfig("full"))
        self.assertTrue((p.loc[g>0,"target_hour"]==p.loc[g>0,"sample_hour"]).all())
        np.testing.assert_array_equal(p.loc[g>0,"scale"],1.)

    def test_budget_per_day_month_and_combined(self):
        h,g,d=example_weather()
        a=make_sampling_plan(h,g,d,SamplingConfig("representative_hours",4,3))
        self.assertLessEqual(a.sample_hour.nunique(),60*4)
        a=make_sampling_plan(h,g,d,SamplingConfig("combined",4,3))
        self.assertLessEqual(a.sample_hour.nunique(),3*3*4)

    def test_dark_day_and_invalid_input(self):
        h,g,d=example_weather(2);g[:24]=0;d[:24]=0
        p=make_sampling_plan(h,g,d,SamplingConfig("combined"))
        self.assertTrue(p.iloc[:24].sample_hour.isna().all())
        with self.assertRaises(ValueError):make_sampling_plan(h,-g,d,SamplingConfig())
        with self.assertRaises(ValueError):make_sampling_plan(h,g,d,SamplingConfig(hours_per_day=0))

    def test_plot_rotation_preserves_area_and_scan(self):
        for az in (90.,180.,240.,270.):
            for kind in ("semi_transparent","conventional"):
                grid=design_grid(Design(azimuth=az),MODULE,ARRAY,kind,nx=3,ny=4)
                p=grid.points();scan,area=grid.scan()
                self.assertAlmostEqual(p.cell_area_m2.sum(),grid.area_m2)
                generated=[]
                for i in range(grid.nx):
                    for j in range(grid.ny):
                        generated.append([scan['xstart']+i*scan['sx_xinc']+j*scan['xinc'],
                                          scan['ystart']+i*scan['sx_yinc']+j*scan['yinc']])
                np.testing.assert_allclose(generated,p[['x_m','y_m']],atol=1e-10)

class ObjectiveTests(unittest.TestCase):
    def test_pareto_dominance_is_not_scalarised(self):
        values=np.array([[1.7,1000.],[1.2,3000.],[1.1,900.]])
        feasible=np.array([True,True,True])
        keep=[]
        for i,v in enumerate(values):
            dominated=feasible & np.all(values>=v,axis=1) & np.any(values>v,axis=1)
            keep.append(not dominated.any())
        self.assertEqual(keep,[True,True,False])

    def test_infeasible_high_metrics_do_not_enter_front(self):
        values=np.array([[1.2,1000.],[99.,999999.]])
        feasible=np.array([True,False])
        dominated=feasible & np.all(values>=values[0],axis=1) & np.any(values>values[0],axis=1)
        self.assertFalse(dominated.any())

    def test_one_harvest_peak_power_and_height_cost(self):
        kind='semi_transparent';config=EngineConfig()
        h=pd.DataFrame({f'{kind}_array_front_effective_incident_energy_Wh_hour':[1000.,2000.],
                        f'{kind}_array_rear_effective_incident_energy_Wh_hour':[0.,0.]})
        grid=design_grid(Design(),MODULE,ARRAY,kind)
        r=economic_metrics(kind,h,[.1,-.05],Design(),MODULE,ARRAY,grid,8.,10.,True,1200000.,config)
        self.assertAlmostEqual(r['crop_revenue_eur'],8*grid.area_m2/10000*.97*220)
        self.assertAlmostEqual(r['module_peak_w'],144*.182*.091*1000*.228)
        higher=Design(clearance_height=5.)
        r2=economic_metrics(kind,h,[.1,-.05],higher,MODULE,ARRAY,grid,8.,10.,True,1200000.,config)
        self.assertGreater(r2['capex_total_eur'],r['capex_total_eur'])
        r3=economic_metrics(kind,h,[.1,-.05],Design(),MODULE,ARRAY,grid,None,10.,False,1200000.,config)
        self.assertFalse(r3['feasible']);self.assertIn('crop_not_mature',r3['infeasible_reason'])

    def test_price_hour_column_rejected(self):
        with TemporaryDirectory() as tmp:
            pd.DataFrame({'hour':np.arange(8760)[::-1],'price_eur_per_kwh':.1}).to_csv(
                Path(tmp)/'electricity_price_2001_DE_LU.csv',index=False)
            with self.assertRaises(ValueError):load_price_profile(example_weather(2)[0],tmp,'csv',.1,[2001])

    def test_uploaded_price_profiles_are_complete_and_load_as_typical_year(self):
        hours=pd.date_range('2001-01-01',periods=8760,freq='h',tz='Etc/GMT-1')
        prices=load_price_profile(hours,ROOT/'prices','csv',.08,[2020,2021,2022,2023,2024,2025])
        self.assertEqual(len(prices),8760)
        self.assertTrue(np.isfinite(prices).all())
        self.assertLess(prices.min(),0.) # negative market-price hours are preserved

    def test_error_table_handles_zero_and_pairs(self):
        t=pd.DataFrame([dict(design_id='A',pv_type='semi',electrical_kwh=100.,crop_yield_t_ha=5.,LER=1.,annual_net_eur_ha=0.)])
        a=t.copy();a.electrical_kwh=110.
        result=compare_sampling(a,t)
        self.assertAlmostEqual(result.electrical_kwh_error_pct.iloc[0],10.)
        self.assertTrue(np.isnan(result.annual_net_eur_ha_error_pct.iloc[0]))

class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pymoo'), 'pymoo is installed from requirements.txt in the user environment')
    def test_search_cache_and_pareto_export_with_mock_physics(self):
        h,g,d=example_weather(365)
        cfg=bridge.CropConfig(start='2001-01-01',end='2001-01-02',accept_demo_assumptions=True,allow_missing_rain_for_pp=True)
        Scenario=make_dataclass('Scenario',[(k,type(v)) for k,v in dict(start_date='2001-01-01',end_date='2001-12-31',
            weather_filename='Aachen_PVGIS_TMY.epw',weather_time_basis='local_standard',ground_grid_x=2,ground_grid_y=2).items()])
        sc=Scenario('2001-01-01','2001-12-31','Aachen_PVGIS_TMY.epw','local_standard',2,2)
        Module=make_dataclass('Module',[(k,type(v)) for k,v in vars(MODULE).items()]);mod=Module(**vars(MODULE))
        Array=make_dataclass('Array',[('n_mods',int),('n_rows',int)]);arr=Array(5,5)
        Settings=make_dataclass('Settings',[('ground_sensor_height',float)]);settings=Settings(.05)
        ns=dict(scenario=sc,module_specification=mod,array_specification=arr,SETTINGS=settings,PVDesign=Design)
        weather=pd.DataFrame({'GHI':g,'DHI':d,'_crop_temp':20.,'_crop_dew':10.,'_crop_wind':2.,
                              '_crop_rain':0.,'_crop_rain_period':1},index=h+pd.Timedelta(hours=1))
        def dummy(*a,**kw):return None
        for name in ['make_hourly_sky','create_selected_module','analyze_array','analyze_module',
                     'analyze_ground_hour','trace_direct_ground_irradiation','calculate_hourly_results',
                     'make_module_surface_scans','make_rectangular_surface_scans','module_geometry',
                     'glass_margin_text','add_hazy_glass_material','validate_cell_array_fit']:ns[name]=dummy
        ns['cultivation_period_bounds']=lambda sc:(pd.Timestamp(sc.start_date),pd.Timestamp(sc.end_date))
        ns['br']=SimpleNamespace(RadianceObj=lambda **kw:None)
        def read(*args):
            ns['ALIGNED_HOURLY_WEATHER']=weather
            return h,h,None,{}
        ns['read_hourly_cultivation_weather']=read
        calls=[]
        def simulate(design,sc,module,array,**kw):
            calls.append((design,kw['calculate_linrui']))
            out=pd.DataFrame({'hour_start':[x.isoformat() for x in h]})
            grounds={}
            for kind,grid in kw['ground_grid'].items():
                if kw['constructions_to_run'] and kind not in kw['constructions_to_run']:continue
                out[f'{kind}_array_front_effective_incident_energy_Wh_hour']=g*30
                out[f'{kind}_array_rear_effective_incident_energy_Wh_hour']=g*3
                if kw['calculate_linrui']:
                    rows=[]
                    for day in cfg.dates():
                        irrad=g[h.date==day].sum()*.8
                        rows += [dict(day=day.isoformat(),total_Wh_m2_day=irrad,**p) for p in grid.points().to_dict('records')]
                    grounds[kind]=pd.DataFrame(rows)
            return out,grounds
        ns['model_for_optimization']=simulate
        def mock_crop(frame,config,radiation):
            yield_dm=10000*np.asarray(radiation).mean()/frame.IRRAD.mean()
            return dict(summary=dict(DOM=cfg.dates()[-1],TWSO=yield_dm,TAGP=2*yield_dm,DVS=2.,CTRAT=1.),
                mature=True,daily=pd.DataFrame({'day':cfg.dates()}),weather=pd.DataFrame({'day':cfg.dates()}),
                parameters={},soil={},site={},management=[])
        with TemporaryDirectory() as tmp,patch.object(bridge,'run_crop',side_effect=mock_crop),\
             patch.object(bridge,'version',return_value='6.0.13'):
            root=Path(tmp);(root/'weather').mkdir()
            import shutil
            shutil.copy(ROOT/'weather'/'Aachen_PVGIS_TMY.epw',root/'weather'/'Aachen_PVGIS_TMY.epw')
            ns['PROJECT_ROOT']=root
            counter=[0]
            def new_run(project):
                counter[0]+=1;p=root/f'crop{counter[0]}';p.mkdir();return p
            with patch.object(bridge,'new_run',side_effect=new_run):
                engine=WeightedEngine(ns,cfg,EngineConfig(),SamplingConfig(),Design(pitch=3.,clearance_height=.8))
                table=engine.search(2,1,1,{'pitch':(4.,12.)},Design())
                self.assertEqual(len(table),4)
                count=len(calls)
                engine.export(table)
                again=engine.search(2,1,1,{'pitch':(4.,12.)},Design())
                self.assertEqual(len(calls),count)
                self.assertTrue((root/'results'/'nsga2_representative_hours'/'pareto_front.csv').exists())
                error=engine.validate_finalists(table[table.pareto],1)
                self.assertEqual(len(error),4)

    def test_bridge_baseline_matches_modes_and_rotated_points(self):
        cfg=bridge.CropConfig(start='2001-10-15',end='2001-10-16',accept_demo_assumptions=True)
        grid=design_grid(Design(azimuth=120.),MODULE,ARRAY,'semi_transparent',nx=2,ny=2)
        frame=pd.DataFrame({'day':cfg.dates(),'IRRAD':[1e6,1e6]})
        radiation=pd.DataFrame([dict(day=day,scenario_id='shade',total_Wh_m2_day=200.,**p)
                                for day in cfg.dates() for p in grid.points().to_dict('records')])
        def mock_crop(*args):
            return dict(summary=dict(DOM=cfg.dates()[-1],TWSO=1000.,TAGP=2000.,DVS=2.,CTRAT=1.),
                mature=True,daily=pd.DataFrame({'day':cfg.dates()}),weather=pd.DataFrame({'day':cfg.dates()}),
                parameters={},soil={},site={},management=[])
        with TemporaryDirectory() as tmp,patch.object(bridge,'run_crop',side_effect=mock_crop),\
             patch.object(bridge,'version',return_value='6.0.13'),patch.object(bridge,'new_run',side_effect=lambda p:Path(tmp)):
            baseline,_=bridge.evaluate_crop_yield_for_hao(frame,cfg,grid,scenario_id='natural_light',spatial_mode='area_mean_approximation')
            out,_=bridge.evaluate_crop_yield_for_hao(frame,cfg,grid,radiation,scenario_id='shade',
                provenance={'source':'mock'},spatial_mode='area_mean_approximation',baseline=baseline)
            self.assertEqual(out['relative_yield_to_natural'],1.)

    def test_notebook_tracing_reconstructs_every_crop_day(self):
        notebook=json.loads((ROOT/'agripv_optimizer_fixed.ipynb').read_text())
        code=''.join(notebook['cells'][8]['source'])
        tree=ast.parse(code)
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='calculate_hourly_results')
        h,g,d=example_weather(3)
        met=SimpleNamespace(datetime=h[g>0]+pd.Timedelta(hours=1))
        ns=dict(np=np,pd=pd,Path=Path,SamplingConfig=SamplingConfig,make_sampling_plan=make_sampling_plan,
                SETTINGS=SimpleNamespace(xgap_semi_transparent=0.,hpc=True),
                ALIGNED_HOURLY_WEATHER=pd.DataFrame({'GHI':g,'DHI':d}),
                read_hourly_cultivation_weather=lambda *a:(h,h,met,{}),
                create_selected_module=lambda *a:None,make_hourly_sky=lambda *a:'sky',
                SimulationContext=lambda **kw:SimpleNamespace(**kw))
        class FakeRadiance:
            def __init__(self,name,**kw):self.name=name;self.materialfiles=[]
            def setGround(self,*a):pass
            def makeScene(self,**kw):return None
            def makeOct(self,**kw):return kw['octname']
            def _getradfiles(self,*a):return []
        ns['br']=SimpleNamespace(RadianceObj=FakeRadiance)
        # Sky name includes timeindex, permitting deterministic GHI-linear mock physics.
        def sample_ghi(ctx):
            i=int(ctx.run_id.split('_')[-1]);return g[g>0][i]
        ns['analyze_array']=lambda ctx,*a:(sample_ghi(ctx)*10,0.)
        seen_flags=[]
        ns['analyze_ground_hour']=lambda octfile,*a,**kw:seen_flags.append(kw.get('with_components')) or tuple(np.full(4,g[g>0][int(octfile.split('_')[-2])]*factor) for factor in (.3,.4,.7))
        def build(daily,points,kind,with_components=True):
            return pd.DataFrame([dict(day=day.isoformat(),pv_type=kind,**p,
                                     direct_Wh_m2_day=v[0],diffuse_residual_Wh_m2_day=v[1],total_Wh_m2_day=v[2])
                                 for day,arr in sorted(daily.items()) for p,v in zip(points.to_dict('records'),arr)])
        ns['build_ground_day_frame']=build
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'mock_notebook','exec'),ns)
        grid=design_grid(Design(),MODULE,ARRAY,'semi_transparent',nx=2,ny=2)
        with TemporaryDirectory() as tmp:
            result,grounds,timing=ns['calculate_hourly_results'](Design(),SimpleNamespace(albedo=.2),MODULE,ARRAY,
                'test',Path(tmp),True,True,ground_grid=grid,sampling_config=SamplingConfig('combined'),crop_dates=[h[0].date(),h[-1].date()])
        self.assertEqual(set(seen_flags),{True})
        with TemporaryDirectory() as tmp:
            seen_flags.clear()
            _,lean,_=ns['calculate_hourly_results'](Design(),SimpleNamespace(albedo=.2),MODULE,ARRAY,
                'test2',Path(tmp),True,True,ground_grid=grid,sampling_config=SamplingConfig('combined'),
                crop_dates=[h[0].date(),h[-1].date()],ground_components=False)
        self.assertEqual(set(seen_flags),{False},'ground_components=False must reach analyze_ground_hour')
        self.assertEqual(len(result),72)
        np.testing.assert_allclose(result.semi_transparent_array_front_effective_incident_energy_Wh_hour,g*10,atol=1e-9)
        self.assertEqual(len(grounds['semi_transparent']),2*4)
        expected=pd.Series(g*.7,index=h).groupby(h.date).sum()
        for day,group in grounds['semi_transparent'].groupby('day'):
            np.testing.assert_allclose(group.total_Wh_m2_day,expected.loc[pd.Timestamp(day).date()])

    def test_notebook_parse_and_safe_defaults(self):
        n=json.loads((ROOT/'agripv_optimizer_fixed.ipynb').read_text())
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        panel=next(''.join(c['source']) for c in n['cells'] if 'USER CONTROL PANEL' in ''.join(c['source']))
        self.assertIn('RUN_SEARCH = False',panel)
        self.assertIn('PRICE_MODE = "csv"',panel)
        section_c=next(''.join(c['source']) for c in n['cells'] if 'MOUNTING = ' in ''.join(c['source']))
        self.assertIn('MOUNTING = "fixed"',section_c)  # default stays the v6 fixed-tilt model
        self.assertIn('NSGA-II',''.join(n['cells'][0]['source']))

if __name__=='__main__':unittest.main(verbosity=2)
