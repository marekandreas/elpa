#!/usr/bin/perl -w

use strict;

my %defs = ();
my %uses = ();

my $use_re = qr/^\s*use\s+(\S+)\s*$/;
my $def_re = qr/^\s*module\s+(\S+)\s*$/;

sub add_use {
	my ($file, $module) = @_;
	if (defined($defs{$module}) && $defs{$module} eq $file) {
		# do not add self-dependencies
		return;
	}
	if (!defined($uses{$file})) {
		$uses{$file} = { $module => 1 };
	} else {
		$uses{$file}{$module} = 1;
	}
}

sub add_def {
	my ($file, $module) = @_;
	if (!defined($defs{$module})) {
		$defs{$module} = $file;
		if (defined($uses{$file}) && defined($uses{$file}{$module})) {
			delete $uses{$file}{$module};
		}
	} else {
		die "Module $module both defined in $file, $defs{$module}";
	}
}

foreach my $file (@ARGV) {
	my $re;
	my $add;
	my $object;
	if ($file =~ /(.*).def_mods$/) {
		$re = $def_re;
		$add = \&add_def;
		$object = $1 . ".o";
	} elsif ($file =~ /^(.*).use_mods$/) {
		$re = $use_re;
		$add = \&add_use;
		$object = $1 . ".o";
	} else {
		die "Unrecognized file extension for '$file'";
	}
	open(FILE,"<",$file) || die "\nCan't open $file: $!\n\n";
	while(<FILE>) {
		chomp;
		$_ = lc($_);
		if ($_ =~ $re) {
			&$add($object, $1);
		} else {
			die "Cannot parse module statement '$_', was expecting $re";
		}
	}
	close(FILE)
}

foreach my $object (sort keys %uses) {
	print "$object: ", join(" ", sort (map {$defs{$_}} keys $uses{$object})), "\n";
}
