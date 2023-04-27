import os

from .util import DRIVER_DIR


PORTFOLIO_DIR = os.path.join(DRIVER_DIR, "portfolios")

ALIASES = {}


ALIASES["seq-sat-fd-autotune-1"] = [
    "--evaluator", "hff=ff(transform=adapt_costs(one))",
    "--evaluator", "hcea=cea()",
    "--evaluator", "hcg=cg(transform=adapt_costs(plusone))",
    "--evaluator", "hgc=goalcount()",
    "--evaluator", "hAdd=add()",
    "--search", """iterated([
lazy(alt([single(sum([g(),weight(hff,10)])),
          single(sum([g(),weight(hff,10)]),pref_only=true)],
         boost=2000),
     preferred=[hff],reopen_closed=false,cost_type=one),
lazy(alt([single(sum([g(),weight(hAdd,7)])),
          single(sum([g(),weight(hAdd,7)]),pref_only=true),
          single(sum([g(),weight(hcg,7)])),
          single(sum([g(),weight(hcg,7)]),pref_only=true),
          single(sum([g(),weight(hcea,7)])),
          single(sum([g(),weight(hcea,7)]),pref_only=true),
          single(sum([g(),weight(hgc,7)])),
          single(sum([g(),weight(hgc,7)]),pref_only=true)],
         boost=1000),
     preferred=[hcea,hgc],reopen_closed=false,cost_type=one),
lazy(alt([tiebreaking([sum([g(),weight(hAdd,3)]),hAdd]),
          tiebreaking([sum([g(),weight(hAdd,3)]),hAdd],pref_only=true),
          tiebreaking([sum([g(),weight(hcg,3)]),hcg]),
          tiebreaking([sum([g(),weight(hcg,3)]),hcg],pref_only=true),
          tiebreaking([sum([g(),weight(hcea,3)]),hcea]),
          tiebreaking([sum([g(),weight(hcea,3)]),hcea],pref_only=true),
          tiebreaking([sum([g(),weight(hgc,3)]),hgc]),
          tiebreaking([sum([g(),weight(hgc,3)]),hgc],pref_only=true)],
         boost=5000),
     preferred=[hcea,hgc],reopen_closed=false,cost_type=normal),
eager(alt([tiebreaking([sum([g(),weight(hAdd,10)]),hAdd]),
           tiebreaking([sum([g(),weight(hAdd,10)]),hAdd],pref_only=true),
           tiebreaking([sum([g(),weight(hcg,10)]),hcg]),
           tiebreaking([sum([g(),weight(hcg,10)]),hcg],pref_only=true),
           tiebreaking([sum([g(),weight(hcea,10)]),hcea]),
           tiebreaking([sum([g(),weight(hcea,10)]),hcea],pref_only=true),
           tiebreaking([sum([g(),weight(hgc,10)]),hgc]),
           tiebreaking([sum([g(),weight(hgc,10)]),hgc],pref_only=true)],
          boost=500),
      preferred=[hcea,hgc],reopen_closed=true,cost_type=normal)
],repeat_last=true,continue_on_fail=true)"""]

ALIASES["seq-sat-fd-autotune-2"] = [
    "--evaluator", "hcea=cea(transform=adapt_costs(plusone))",
    "--evaluator", "hcg=cg(transform=adapt_costs(one))",
    "--evaluator", "hgc=goalcount(transform=adapt_costs(plusone))",
    "--evaluator", "hff=ff()",
    "--search", """iterated([
ehc(hcea,preferred=[hcea],preferred_usage=0,cost_type=normal),
lazy(alt([single(sum([weight(g(),2),weight(hff,3)])),
          single(sum([weight(g(),2),weight(hff,3)]),pref_only=true),
          single(sum([weight(g(),2),weight(hcg,3)])),
          single(sum([weight(g(),2),weight(hcg,3)]),pref_only=true),
          single(sum([weight(g(),2),weight(hcea,3)])),
          single(sum([weight(g(),2),weight(hcea,3)]),pref_only=true),
          single(sum([weight(g(),2),weight(hgc,3)])),
          single(sum([weight(g(),2),weight(hgc,3)]),pref_only=true)],
         boost=200),
     preferred=[hcea,hgc],reopen_closed=false,cost_type=one),
lazy(alt([single(sum([g(),weight(hff,5)])),
          single(sum([g(),weight(hff,5)]),pref_only=true),
          single(sum([g(),weight(hcg,5)])),
          single(sum([g(),weight(hcg,5)]),pref_only=true),
          single(sum([g(),weight(hcea,5)])),
          single(sum([g(),weight(hcea,5)]),pref_only=true),
          single(sum([g(),weight(hgc,5)])),
          single(sum([g(),weight(hgc,5)]),pref_only=true)],
         boost=5000),
     preferred=[hcea,hgc],reopen_closed=true,cost_type=normal),
lazy(alt([single(sum([g(),weight(hff,2)])),
          single(sum([g(),weight(hff,2)]),pref_only=true),
          single(sum([g(),weight(hcg,2)])),
          single(sum([g(),weight(hcg,2)]),pref_only=true),
          single(sum([g(),weight(hcea,2)])),
          single(sum([g(),weight(hcea,2)]),pref_only=true),
          single(sum([g(),weight(hgc,2)])),
          single(sum([g(),weight(hgc,2)]),pref_only=true)],
         boost=1000),
     preferred=[hcea,hgc],reopen_closed=true,cost_type=one)
],repeat_last=true,continue_on_fail=true)"""]

def _get_lama(**kwargs):
    return [
        "--if-unit-cost",
        "--evaluator",
        "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),pref={pref})".format(**kwargs),
        "--evaluator", "hff=ff()",
        "--search", """iterated([
                         lazy_greedy([hff,hlm],preferred=[hff,hlm]),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=5),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=3),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=2),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=1)
                         ],repeat_last=true,continue_on_fail=true)""",
        "--if-non-unit-cost",
        "--evaluator",
        "hlm1=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref={pref})".format(**kwargs),
        "--evaluator", "hff1=ff(transform=adapt_costs(one))",
        "--evaluator",
        "hlm2=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(plusone),pref={pref})".format(**kwargs),
        "--evaluator", "hff2=ff(transform=adapt_costs(plusone))",
        "--search", """iterated([
                         lazy_greedy([hff1,hlm1],preferred=[hff1,hlm1],
                                     cost_type=one,reopen_closed=false),
                         lazy_greedy([hff2,hlm2],preferred=[hff2,hlm2],
                                     reopen_closed=false),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=5),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=3),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=2),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=1)
                         ],repeat_last=true,continue_on_fail=true)""",
        # Append --always to be on the safe side if we want to append
        # additional options later.
        "--always"]

ALIASES["seq-sat-lama-2011"] = _get_lama(pref="true")
ALIASES["lama"] = _get_lama(pref="false")

ALIASES["lama-first"] = [
    "--evaluator",
    "hlm=lmcount(lm_factory=lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false)",
    "--evaluator", "hff=ff(transform=adapt_costs(one))",
    "--search", """lazy_greedy([hff,hlm],preferred=[hff,hlm],
                               cost_type=one,reopen_closed=false)"""]

ALIASES["seq-opt-bjolp"] = [
    "--evaluator",
    "lmc=lmcount(lm_merged([lm_rhw(),lm_hm(m=1)]),admissible=true)",
    "--search",
    "astar(lmc,lazy_evaluator=lmc)"]

ALIASES["seq-opt-lmcut"] = [
    "--search", "astar(lmcut())"]


ALIASES["seq-sat-fdss-2018-0"] = [
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(hff),single(hff,pref_only=true),single(hlm),single(hlm,pref_only=true),type_based([hff,g()])],boost=1000),preferred=[hff,hlm],cost_type=one,reopen_closed=false,randomize_successors=true,preferred_successors_first=false)"]

ALIASES["seq-sat-fdss-2018-1"] = [
    "--landmarks",
    "lmg=lm_rhw(only_causal_landmarks=false,disjunctive_landmarks=true,use_orders=false)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([type_based([g()]),single(hlm),single(hlm,pref_only=true),single(hff),single(hff,pref_only=true)],boost=0),preferred=[hlm],reopen_closed=false,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-2"] = [
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(hff),single(hff,pref_only=true),single(hlm),single(hlm,pref_only=true)],boost=1000),preferred=[hff,hlm],cost_type=one,reopen_closed=false,randomize_successors=false,preferred_successors_first=true)"]

ALIASES["seq-sat-fdss-2018-3"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "eager_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-4"] = [
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(hff),single(hff,pref_only=true),single(hlm),single(hlm,pref_only=true)],boost=1000),preferred=[hff,hlm],cost_type=one,reopen_closed=false,randomize_successors=true,preferred_successors_first=true)"]

ALIASES["seq-sat-fdss-2018-5"] = [
    "--landmarks",
    "lmg=lm_rhw(only_causal_landmarks=false,disjunctive_landmarks=true,use_orders=false)",
    "--evaluator",
    "hcg=cg(transform=adapt_costs(plusone))",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([single(sum([g(),weight(hlm,10)])),single(sum([g(),weight(hlm,10)]),pref_only=true),single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true),single(sum([g(),weight(hcg,10)])),single(sum([g(),weight(hcg,10)]),pref_only=true)],boost=1000),preferred=[hlm,hcg],reopen_closed=false,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-6"] = [
    "--evaluator",
    "hcea=cea(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "lazy_greedy([hcea,hlm],preferred=[hcea,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-7"] = [
    "--evaluator",
    "hadd=add(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "lazy(alt([type_based([g()]),single(hadd),single(hadd,pref_only=true),single(hlm),single(hlm,pref_only=true)]),preferred=[hadd,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-8"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true)],boost=2000),preferred=[hff],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-9"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "eager(alt([type_based([g()]),single(hcg),single(hcg,pref_only=true),single(hlm),single(hlm,pref_only=true)]),preferred=[hcg,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-10"] = [
    "--landmarks",
    "lmg=lm_rhw(only_causal_landmarks=false,disjunctive_landmarks=true,use_orders=true)",
    "--evaluator",
    "hcea=cea(transform=adapt_costs(plusone))",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([single(hlm),single(hlm,pref_only=true),single(hff),single(hff,pref_only=true),single(hcea),single(hcea,pref_only=true)],boost=0),preferred=[hlm,hcea],reopen_closed=false,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-11"] = [
    "--evaluator",
    "hcea=cea(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "lazy_wastar([hcea,hlm],w=3,preferred=[hcea,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-12"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true),single(sum([g(),weight(hcg,10)])),single(sum([g(),weight(hcg,10)]),pref_only=true)],boost=100),preferred=[hcg],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-13"] = [
    "--evaluator",
    "hgoalcount=goalcount(transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff()",
    "--search",
    "lazy(alt([single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true),single(sum([g(),weight(hgoalcount,10)])),single(sum([g(),weight(hgoalcount,10)]),pref_only=true)],boost=2000),preferred=[hff,hgoalcount],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-14"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "eager(alt([type_based([g()]),single(sum([g(),weight(hff,3)])),single(sum([g(),weight(hff,3)]),pref_only=true),single(sum([g(),weight(hlm,3)])),single(sum([g(),weight(hlm,3)]),pref_only=true)]),preferred=[hff,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-15"] = [
    "--landmarks",
    "lmg=lm_rhw(only_causal_landmarks=false,disjunctive_landmarks=false,use_orders=true)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=false,transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--evaluator",
    "hblind=blind()",
    "--search",
    "lazy(alt([type_based([g()]),single(sum([g(),weight(hblind,2)])),single(sum([g(),weight(hblind,2)]),pref_only=true),single(sum([g(),weight(hlm,2)])),single(sum([g(),weight(hlm,2)]),pref_only=true),single(sum([g(),weight(hff,2)])),single(sum([g(),weight(hff,2)]),pref_only=true)],boost=4419),preferred=[hlm],reopen_closed=true,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-16"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy_wastar([hff],w=3,preferred=[hff],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-17"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([type_based([g()]),single(hcg),single(hcg,pref_only=true)],boost=0),preferred=[hcg],reopen_closed=true,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-18"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one))",
    "--search",
    "lazy(alt([type_based([g()]),single(sum([g(),weight(hcg,3)])),single(sum([g(),weight(hcg,3)]),pref_only=true),single(sum([g(),weight(hlm,3)])),single(sum([g(),weight(hlm,3)]),pref_only=true)]),preferred=[hcg,hlm],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-19"] = [
    "--evaluator",
    "hcea=cea(transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--evaluator",
    "hblind=blind()",
    "--search",
    "eager(alt([single(sum([g(),weight(hblind,10)])),single(sum([g(),weight(hblind,10)]),pref_only=true),single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true),single(sum([g(),weight(hcea,10)])),single(sum([g(),weight(hcea,10)]),pref_only=true)],boost=536),preferred=[hff],reopen_closed=false)"]

ALIASES["seq-sat-fdss-2018-20"] = [
    "--evaluator",
    "hcea=cea(transform=adapt_costs(one))",
    "--search",
    "eager_greedy([hcea],preferred=[hcea],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-21"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "eager(alt([single(sum([g(),weight(hff,3)])),single(sum([g(),weight(hff,3)]),pref_only=true)]),preferred=[hff],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-22"] = [
    "--evaluator",
    "hgoalcount=goalcount(transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hcg=cg()",
    "--search",
    "lazy(alt([type_based([g()]),single(sum([weight(g(),2),weight(hblind,3)])),single(sum([weight(g(),2),weight(hblind,3)]),pref_only=true),single(sum([weight(g(),2),weight(hff,3)])),single(sum([weight(g(),2),weight(hff,3)]),pref_only=true),single(sum([weight(g(),2),weight(hcg,3)])),single(sum([weight(g(),2),weight(hcg,3)]),pref_only=true),single(sum([weight(g(),2),weight(hgoalcount,3)])),single(sum([weight(g(),2),weight(hgoalcount,3)]),pref_only=true)],boost=3662),preferred=[hff],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-23"] = [
    "--evaluator",
    "hgoalcount=goalcount(transform=adapt_costs(one))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hcg=cg()",
    "--search",
    "lazy(alt([single(sum([weight(g(),2),weight(hblind,3)])),single(sum([weight(g(),2),weight(hblind,3)]),pref_only=true),single(sum([weight(g(),2),weight(hff,3)])),single(sum([weight(g(),2),weight(hff,3)]),pref_only=true),single(sum([weight(g(),2),weight(hcg,3)])),single(sum([weight(g(),2),weight(hcg,3)]),pref_only=true),single(sum([weight(g(),2),weight(hgoalcount,3)])),single(sum([weight(g(),2),weight(hgoalcount,3)]),pref_only=true)],boost=3662),preferred=[hff],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-24"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([single(sum([g(),weight(hcg,10)])),single(sum([g(),weight(hcg,10)]),pref_only=true)],boost=0),preferred=[hcg],reopen_closed=false,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-25"] = [
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--search",
    "eager(alt([single(sum([g(),weight(hcg,3)])),single(sum([g(),weight(hcg,3)]),pref_only=true)]),preferred=[hcg],cost_type=one)"]

ALIASES["seq-sat-fdss-2018-26"] = [
    "--landmarks",
    "lmg=lm_reasonable_orders_hps(lm_rhw(only_causal_landmarks=true,disjunctive_landmarks=true,use_orders=true))",
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hadd=add()",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=false,pref=true,transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff()",
    "--search",
    "lazy(alt([single(sum([weight(g(),2),weight(hblind,3)])),single(sum([weight(g(),2),weight(hblind,3)]),pref_only=true),single(sum([weight(g(),2),weight(hff,3)])),single(sum([weight(g(),2),weight(hff,3)]),pref_only=true),single(sum([weight(g(),2),weight(hlm,3)])),single(sum([weight(g(),2),weight(hlm,3)]),pref_only=true),single(sum([weight(g(),2),weight(hadd,3)])),single(sum([weight(g(),2),weight(hadd,3)]),pref_only=true)],boost=2474),preferred=[hadd],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-27"] = [
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hadd=add()",
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--evaluator",
    "hhmax=hmax()",
    "--search",
    "eager(alt([tiebreaking([sum([g(),weight(hblind,7)]),hblind]),tiebreaking([sum([g(),weight(hhmax,7)]),hhmax]),tiebreaking([sum([g(),weight(hadd,7)]),hadd]),tiebreaking([sum([g(),weight(hcg,7)]),hcg])],boost=2142),preferred=[],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-28"] = [
    "--evaluator",
    "hadd=add(transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff()",
    "--search",
    "lazy(alt([tiebreaking([sum([weight(g(),4),weight(hff,5)]),hff]),tiebreaking([sum([weight(g(),4),weight(hff,5)]),hff],pref_only=true),tiebreaking([sum([weight(g(),4),weight(hadd,5)]),hadd]),tiebreaking([sum([weight(g(),4),weight(hadd,5)]),hadd],pref_only=true)],boost=2537),preferred=[hff,hadd],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-29"] = [
    "--landmarks",
    "lmg=lm_hm(conjunctive_landmarks=false,use_orders=false,m=1)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,transform=transform=adapt_costs(plusone))",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([type_based([g()]),single(hlm),single(hlm,pref_only=true),single(hff),single(hff,pref_only=true)],boost=5000),preferred=[hlm],reopen_closed=false)"]

ALIASES["seq-sat-fdss-2018-30"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "lazy(alt([single(sum([weight(g(),2),weight(hff,3)])),single(sum([weight(g(),2),weight(hff,3)]),pref_only=true)],boost=5000),preferred=[hff],reopen_closed=true,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-31"] = [
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hff=ff(transform=adapt_costs(one))",
    "--search",
    "eager(alt([single(sum([g(),weight(hblind,2)])),single(sum([g(),weight(hff,2)]))],boost=4480),preferred=[],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-32"] = [
    "--landmarks",
    "lmg=lm_hm(conjunctive_landmarks=false,use_orders=false,m=1)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true)",
    "--evaluator",
    "hff=ff()",
    "--search",
    "lazy(alt([type_based([g()]),single(hlm),single(hlm,pref_only=true),single(hff),single(hff,pref_only=true)],boost=1000),preferred=[hlm,hff],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-33"] = [
    "--landmarks",
    "lmg=lm_hm(conjunctive_landmarks=true,use_orders=true,m=1)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true)",
    "--evaluator",
    "hff=ff()",
    "--search",
    "lazy(alt([tiebreaking([sum([g(),weight(hlm,10)]),hlm]),tiebreaking([sum([g(),weight(hlm,10)]),hlm],pref_only=true),tiebreaking([sum([g(),weight(hff,10)]),hff]),tiebreaking([sum([g(),weight(hff,10)]),hff],pref_only=true)],boost=200),preferred=[hlm],reopen_closed=true,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-34"] = [
    "--landmarks",
    "lmg=lm_hm(conjunctive_landmarks=false,use_orders=false,m=1)",
    "--evaluator",
    "hcg=cg(transform=adapt_costs(one))",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true)",
    "--search",
    "lazy(alt([single(hlm),single(hlm,pref_only=true),single(hcg),single(hcg,pref_only=true)],boost=0),preferred=[hcg],reopen_closed=false,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-35"] = [
    "--landmarks",
    "lmg=lm_exhaust(only_causal_landmarks=false)",
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--evaluator",
    "hhmax=hmax()",
    "--evaluator",
    "hblind=blind()",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,pref=false,transform=adapt_costs(one))",
    "--search",
    "lazy(alt([type_based([g()]),single(sum([g(),weight(hblind,3)])),single(sum([g(),weight(hblind,3)]),pref_only=true),single(sum([g(),weight(hff,3)])),single(sum([g(),weight(hff,3)]),pref_only=true),single(sum([g(),weight(hlm,3)])),single(sum([g(),weight(hlm,3)]),pref_only=true),single(sum([g(),weight(hhmax,3)])),single(sum([g(),weight(hhmax,3)]),pref_only=true)],boost=3052),preferred=[hff],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-36"] = [
    "--evaluator",
    "hff=ff(transform=adapt_costs(plusone))",
    "--search",
    "lazy(alt([tiebreaking([sum([g(),hff]),hff]),tiebreaking([sum([g(),hff]),hff],pref_only=true)],boost=432),preferred=[hff],reopen_closed=true,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-37"] = [
    "--landmarks",
    "lmg=lm_merged([lm_rhw(only_causal_landmarks=false,disjunctive_landmarks=false,use_orders=true),lm_hm(m=1,conjunctive_landmarks=true,use_orders=true)])",
    "--evaluator",
    "hff=ff()",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true)",
    "--search",
    "lazy(alt([single(sum([g(),weight(hff,10)])),single(sum([g(),weight(hff,10)]),pref_only=true),single(sum([g(),weight(hlm,10)])),single(sum([g(),weight(hlm,10)]),pref_only=true)],boost=500),preferred=[hff],reopen_closed=false,cost_type=plusone)"]

ALIASES["seq-sat-fdss-2018-38"] = [
    "--landmarks",
    "lmg=lm_exhaust(only_causal_landmarks=false)",
    "--evaluator",
    "hgoalcount=goalcount(transform=adapt_costs(plusone))",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=false)",
    "--evaluator",
    "hff=ff()",
    "--evaluator",
    "hblind=blind()",
    "--search",
    "eager(alt([tiebreaking([sum([weight(g(),8),weight(hblind,9)]),hblind]),tiebreaking([sum([weight(g(),8),weight(hlm,9)]),hlm]),tiebreaking([sum([weight(g(),8),weight(hff,9)]),hff]),tiebreaking([sum([weight(g(),8),weight(hgoalcount,9)]),hgoalcount])],boost=2005),preferred=[],reopen_closed=true)"]

ALIASES["seq-sat-fdss-2018-39"] = [
    "--landmarks",
    "lmg=lm_zg(use_orders=false)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true,pref=false)",
    "--search",
    "eager(single(sum([g(),weight(hlm,3)])),preferred=[],reopen_closed=true,cost_type=one)"]

ALIASES["seq-sat-fdss-2018-40"] = [
    "--landmarks",
    "lmg=lm_hm(conjunctive_landmarks=true,use_orders=false,m=1)",
    "--evaluator",
    "hlm=lmcount(lmg,admissible=true)",
    "--search",
    "eager(single(sum([g(),weight(hlm,5)])),preferred=[],reopen_closed=true,cost_type=one)"]



PORTFOLIOS = {}
for portfolio in os.listdir(PORTFOLIO_DIR):
    if portfolio == "__pycache__":
        continue
    name, ext = os.path.splitext(portfolio)
    assert ext == ".py", portfolio
    PORTFOLIOS[name.replace("_", "-")] = os.path.join(PORTFOLIO_DIR, portfolio)


def show_aliases():
    for alias in sorted(list(ALIASES) + list(PORTFOLIOS)):
        print(alias)


def set_options_for_alias(alias_name, args):
    """
    If alias_name is an alias for a configuration, set args.search_options
    to the corresponding command-line arguments. If it is an alias for a
    portfolio, set args.portfolio to the path to the portfolio file.
    Otherwise raise KeyError.
    """
    assert not args.search_options
    assert not args.portfolio

    if alias_name in ALIASES:
        args.search_options = [x.replace(" ", "").replace("\n", "")
                               for x in ALIASES[alias_name]]
    elif alias_name in PORTFOLIOS:
        args.portfolio = PORTFOLIOS[alias_name]
    else:
        raise KeyError(alias_name)
