_f90_verbose = $(_f90_verbose_$(V))
_f90_verbose_ = $(_f90_verbose_$(AM_DEFAULT_VERBOSITY))
_f90_verbose_0 = @echo "  $1";
_f90_targets = $(subst -,_,$(patsubst %.la,%_la,$(patsubst %.a,%_a,$(patsubst %.so,%_so,$(PROGRAMS) $(LTLIBRARIES)))))
FORTRAN_CPP ?= cpp -P -traditional -Wall -Werror

# $1 source files
#
# returns: file without any .F90 .f90 .F .f extension
define strip_fortran_ext
$(patsubst %.F90,%,$(patsubst %.f90,%,$(patsubst %.F,%,$(patsubst %.f,%,$1))))
endef

# $1 program
#
# returns:
#  '1' if object files for target $1 are prefixed due to 'per-target' flags,
#  '' (the empty string) otherwise. See the automake manual for 'per-target'
#  compilation
#
define is_per_target
$(if $(filter $(call strip_fortran_ext,$(firstword $(call fortran_sources,$1))),$(patsubst %.o,%,$(patsubst %.lo,%,$($1_OBJECTS)))),,1)
endef

# $1 top-level target name (i.e. an entry of _f90_targets)
#
# returns: all target source files matching *.F90 *.f90 *.F *.f
define fortran_sources
$(filter %.F90 %.f90 %.F %.f,$($1_SOURCES))
endef

# $1 top-level target name
#
# returns: the appropriate extension (i.e. 'o' for normal programs, '.lo' for libraries)
define object_extension
$(if $(filter $1,$(PROGRAMS)),o,lo)
endef

# $1 source_file
# $2 stem
# $3 program
define module_targets
$(eval _$3_use_mods += $(dir $1)$2$(call strip_fortran_ext,$(notdir $1)).use_mods.$3.$(call object_extension,$3))
$(dir $1)$2$(call strip_fortran_ext,$(notdir $1)).use_mods.$3.$(call object_extension,$3): $1 $(dir $1)$(am__dirstamp)
	$(call _f90_verbose,F90 USE  [$3] $$<)$(FORTRAN_CPP) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $($p_CPPFLAGS) $(CPPFLAGS) -o /dev/stdout $$< | grep -i -o '^ *use [^ ,!:]*' | sort -u > $$@

$(eval _$3_def_mods += $(dir $1)$2$(call strip_fortran_ext,$(notdir $1)).def_mods.$3.$(call object_extension,$3))
$(dir $1)$2$(call strip_fortran_ext,$(notdir $1)).def_mods.$3.$(call object_extension,$3): $1 $(dir $1)$(am__dirstamp)
	$(call _f90_verbose,F90 MOD  [$3] $$<)$(FORTRAN_CPP) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $($p_CPPFLAGS) $(CPPFLAGS) -o /dev/stdout $$< | grep -i -o '^ *module [^!]*' | grep -v "\<procedure\>" > $$@ || true

endef
$(foreach p,$(_f90_targets),$(if $(call is_per_target,$p),$(foreach s,$(call fortran_sources,$p),$(eval $(call module_targets,$s,$p-,$p))),$(foreach s,$(call fortran_sources,$p),$(eval $(call module_targets,$s,,$p)))))

_f90_depdir=$(abs_builddir)/.fortran_dependencies
_f90_depfile = $(_f90_depdir)/dependencies.mk

define is_clean
$(if $(filter-out mostlyclean clean distclean maintainer-clean,$(MAKECMDGOALS)),0,1)
endef

define _fdep_newline


endef

ifneq ($(call is_clean),1)
include $(_f90_depfile)
endif
$(_f90_depfile): $(top_srcdir)/fdep/fortran_dependencies.pl $(foreach p,$(_f90_targets),$(_$p_use_mods) $(_$p_def_mods)) | $(foreach p,$(_f90_targets),$(_f90_depdir)/$p)
	$(call _f90_verbose,F90 DEPS $@)echo > $@; $(foreach p,$(_f90_targets),$(top_srcdir)/fdep/fortran_dependencies.pl $p $(_$p_use_mods) $(_$p_def_mods) >> $@ || { rm $@; exit 1; } ;$(_fdep_newline))

$(_f90_depdir):
	@mkdir $@

$(foreach p,$(_f90_targets),$(_f90_depdir)/$p): | $(_f90_depdir)
	@mkdir $@

CLEANFILES += $(foreach p,$(_f90_targets),$(_$p_def_mods) $(_$p_use_mods))
CLEANFILES += $(foreach p,$(_f90_targets),$(_f90_depdir)/$p/*)
CLEANFILES += $(_f90_depfile)
